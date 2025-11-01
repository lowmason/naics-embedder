import io
from dataclasses import asdict, dataclass, field
import json
import time
from typing import Dict, Mapping, Optional
import logging

import httpx
import polars as pl
from polars._typing import PolarsDataType
from rich.progress import track

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class Config:
    """Configuration for downloading and preprocessing NAICS data."""
    output_parquet: str = './data/naics_descriptions.parquet'

    url_codes: str = 'https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx'
    url_index: str = 'https://www.census.gov/naics/2022NAICS/2022_NAICS_Index_File.xlsx'
    url_descriptions: str = 'https://www.census.gov/naics/2022NAICS/2022_NAICS_Descriptions.xlsx'
    url_exclusions: str = 'https://www.census.gov/naics/2022NAICS/2022_NAICS_Cross_References.xlsx'

    sheet_codes: str = 'tbl_2022_title_description_coun'
    sheet_index: str = '2022NAICS'
    sheet_descriptions: str = '2022_NAICS_Descriptions'
    sheet_exclusions: str = '2022_NAICS_Cross_References'
    schema_codes: Mapping[str, PolarsDataType] = field(
        default_factory=lambda: {
            "Seq. No.": "UInt32",
            "2022 NAICS US   Code": "String",
            "2022 NAICS US Title": "String",
        }
    )
    schema_index: Mapping[str, PolarsDataType] = field(
        default_factory=lambda: {
            "NAICS22": "String",
            "INDEX ITEM": "String",
        }
    )
    schema_descriptions: Mapping[str, PolarsDataType] = field(
        default_factory=lambda: {
            "NAICS": "String",
            "DESCRIPTION": "String",
        }
    )
    schema_exclusions: Mapping[str, PolarsDataType] = field(
        default_factory=lambda: {
            "NAICS": "String",
            "CROSS-REFERENCE": "String",
        }
    )


# -------------------------------------------------------------------------------------------------
# 1. Download
# -------------------------------------------------------------------------------------------------

def _download_xlsx(url: str) -> io.BytesIO:
    """Helper to download an Excel file into an in-memory buffer."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()  # Raise an exception for bad status codes
            return io.BytesIO(response.content)
    except httpx.RequestError as e:
        logger.error(f"Error downloading {url}: {e}")
        raise

def download_files(cfg: Config) -> Dict[str, io.BytesIO]:
    """Downloads all required NAICS Excel files from Census.gov."""
    urls = {
        'codes': cfg.url_codes,
        'index': cfg.url_index,
        'descriptions': cfg.url_descriptions,
        'exclusions': cfg.url_exclusions,
    }
    
    downloaded_files = {}
    
    desc = "Downloading NAICS data"
    for name, url in track(urls.items(), description=desc):
        logger.info(f"Downloading {name} data from {url}...")
        downloaded_files[name] = _download_xlsx(url)
        time.sleep(0.5) # Be polite
        
    logger.info("All files downloaded successfully.")
    return downloaded_files


# -------------------------------------------------------------------------------------------------
# 2. Preprocess
# -------------------------------------------------------------------------------------------------

def _read_excel(
    file_buffer: io.BytesIO,
    sheet_name: str,
    schema: Mapping[str, PolarsDataType],
) -> pl.DataFrame:
    """Helper to read a specific sheet from an in-memory Excel file."""
    return pl.read_excel(
        file_buffer,
        sheet_name=sheet_name,
        schema_overrides=schema,
        read_options={"has_header": True},
    )

def get_titles(
    file_buffer: io.BytesIO, 
    sheet_name: str, 
    schema: Mapping[str, PolarsDataType]
) -> pl.DataFrame:
    """Extracts and cleans NAICS codes and titles."""
    return (
        _read_excel(file_buffer, sheet_name, schema)
        .select(
            code=pl.col("2022 NAICS US   Code").str.replace_all("-", ""),
            title=pl.col("2022 NAICS US Title"),
        )
        .filter(pl.col("code").str.lengths() >= 2)
        .with_columns(
            level=pl.col("code").str.lengths(),
            index=pl.col("Seq. No.").cum_count().sub(1),
        )
        .select("index", "level", "code", "title")
        .sort("index")
    )

def get_examples(
    file_buffer: io.BytesIO, 
    sheet_name: str, 
    schema: Mapping[str, PolarsDataType]
) -> pl.DataFrame:
    """Extracts and groups illustrative examples for each code."""
    return (
        _read_excel(file_buffer, sheet_name, schema)
        .select(
            code=pl.col("NAICS22").str.replace_all("-", ""),
            example=pl.col("INDEX ITEM").str.to_lowercase(),
        )
        .filter(pl.col("code").str.lengths() >= 2)
        .group_by("code")
        .agg(pl.col("example").alias("examples"))
    )

def get_descriptions(
    file_buffer: io.BytesIO, 
    sheet_name: str, 
    schema: Mapping[str, PolarsDataType]
) -> pl.DataFrame:
    """Extracts and cleans detailed descriptions for each code."""
    return (
        _read_excel(file_buffer, sheet_name, schema)
        .select(
            code=pl.col("NAICS").str.replace_all("-", ""),
            description=pl.col("DESCRIPTION"),
        )
        .filter(pl.col("code").str.lengths() >= 2)
        .with_columns(
            description=pl.col("description")
            .str.replace_all("\n", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip()
        )
    )

def get_exclusions(
    file_buffer: io.BytesIO, 
    sheet_name: str, 
    schema: Mapping[str, PolarsDataType]
) -> pl.DataFrame:
    """Extracts and cleans exclusion (cross-reference) information."""
    return (
        _read_excel(file_buffer, sheet_name, schema)
        .select(
            code=pl.col("NAICS").str.replace_all("-", ""),
            excluded=pl.col("CROSS-REFERENCE"),
        )
        .filter(pl.col("code").str.lengths() >= 2)
        .with_columns(
            excluded=pl.col("excluded")
            .str.replace_all("\n", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip()
        )
        .group_by("code")
        .agg(pl.col("excluded").list().join(separator="; "))
    )

def _fill_missing_descriptions(
    titles: pl.DataFrame, descriptions: pl.DataFrame
) -> pl.DataFrame:
    """
    Fills missing descriptions for higher-level codes by borrowing from 
    their children's descriptions.
    """
    naics_desc_filled = (
        titles.join(descriptions, how="left", on="code")
        .filter(pl.col("description").is_null())
        .select("index", "level", "code")
    )

    naics_desc_complete = (
        titles.join(descriptions, how="inner", on="code")
        .filter(pl.col("description").is_not_null())
        .select("index", "level", "code", "description")
    )

    logger.info(f"Total codes: {titles.height}")
    logger.info(f"Codes with descriptions: {naics_desc_complete.height}")
    logger.info(f"Codes missing descriptions: {naics_desc_filled.height}")

    # Fill level 4
    naics_desc_4_missing = naics_desc_filled.filter(pl.col("level") == 4)
    naics_desc_4_complete = (
        naics_desc_4_missing.join(
            naics_desc_complete.filter(pl.col("level") == 5).select(
                code_5=pl.col("code"), description_5=pl.col("description")
            ),
            how="left",
            on=pl.Expr.register_plugin_function(
                "register_py_mod", "is_descendent", is_entrypoint=True
            )(pl.col("code"), pl.col("code_5")),
        )
        .group_by("index", "level", "code")
        .agg(
            pl.col("description_5")
            .filter(pl.col("description_5").is_not_null())
            .list()
            .join(separator=" ")
            .alias("description")
        )
        .select("index", "level", "code", "description")
    )

    # Fill level 2, 3, 5
    naics_desc_2_3_5_missing = naics_desc_filled.filter(pl.col("level") != 4)
    naics_desc_2_3_5_complete = (
        naics_desc_2_3_5_missing.join(
            naics_desc_complete.filter(pl.col("level") == 6).select(
                code_6=pl.col("code"), description_6=pl.col("description")
            ),
            how="left",
            on=pl.Expr.register_plugin_function(
                "register_py_mod", "is_descendent", is_entrypoint=True
            )(pl.col("code"), pl.col("code_6")),
        )
        .group_by("index", "level", "code")
        .agg(
            pl.col("description_6")
            .filter(pl.col("description_6").is_not_null())
            .list()
            .join(separator=" ")
            .alias("description")
        )
        .select("index", "level", "code", "description")
        .with_columns(
            pl.col("description")
            .str.replace_all("Illustrative Examples: ", " ")
            .str.replace_all("Cross-References: ", " ")
            .str.replace(
                "This U.S. industry", "This industry group", literal=True
            )
            .str.replace("This U.S. industry", "This industry", literal=True)
            .str.replace(
                "This NAICS industry", "This industry group", literal=True
            ),
        )
        .unique(subset=["code"])
    )

    # Combine all descriptions
    naics_descriptions = pl.concat(
        [naics_desc_complete, naics_desc_4_complete, naics_desc_2_3_5_complete]
    )
    
    logger.info(f"Filled {naics_desc_4_complete.height} level 4 descriptions.")
    logger.info(f"Filled {naics_desc_2_3_5_complete.height} other descriptions.")
    logger.info(f"Total descriptions: {naics_descriptions.height}")

    return naics_descriptions


# -------------------------------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------------------------------

def download_naics_data() -> None:
    """
    Main entry point for downloading and preprocessing all NAICS data.
    """
    # Configuration
    cfg = Config()
    logger.info("Configuration:")
    logger.info(json.dumps(asdict(cfg), indent=2))

    # Download
    files = download_files(cfg)

    # Process each file
    logger.info("Processing downloaded files...")
    naics_titles = get_titles(
        files["codes"], cfg.sheet_codes, cfg.schema_codes
    )
    naics_examples = get_examples(
        files["index"], cfg.sheet_index, cfg.schema_index
    )
    naics_descriptions_raw = get_descriptions(
        files["descriptions"], cfg.sheet_descriptions, cfg.schema_descriptions
    )
    naics_exclusions = get_exclusions(
        files["exclusions"], cfg.sheet_exclusions, cfg.schema_exclusions
    )

    # Fill missing descriptions
    logger.info("Filling missing descriptions...")
    naics_descriptions = _fill_missing_descriptions(
        naics_titles, naics_descriptions_raw
    )

    # Join all components
    logger.info("Joining all data components...")
    naics_final = (
        naics_titles.join(naics_descriptions, how="left", on=["index", "level", "code"])
        .join(naics_exclusions, how="left", on="code")
        .join(naics_examples, how="left", on="code")
        .with_columns(
            examples=pl.col("examples").list.join(separator="; ")
        )
        .select("index", "level", "code", "title", "description", "excluded", "examples")
        .sort("index")
    )

    # Write final output
    logger.info(f"Writing final output to {cfg.output_parquet}...")
    naics_final.write_parquet(cfg.output_parquet)

    logger.info(f"\nSuccessfully generated {cfg.output_parquet}")
    logger.info(f"Total codes processed: {naics_final.height}")


if __name__ == "__main__":
    # This script can still be run directly if needed,
    # but it's now designed to be called by cli.py
    configure_logging()
    download_naics_data()
