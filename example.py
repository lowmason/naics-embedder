'''
NAICS Data Preprocessing Pipeline

Downloads and processes 2022 NAICS (North American Industry Classification System) data
from Census Bureau Excel files, normalizing codes, descriptions, examples, and exclusions
into a unified Parquet file.
'''

# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import io
from dataclasses import dataclass, field
from typing import Dict

import httpx
import polars as pl

# -------------------------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------------------------

@dataclass
class Config:

    '''Configuration for NAICS data sources and processing.'''

    url_codes: str = 'https://www.census.gov/naics/2022NAICS/2-6%20digit_2022_Codes.xlsx'
    url_index: str = 'https://www.census.gov/naics/2022NAICS/2022_NAICS_Index_File.xlsx'
    url_descriptions: str = 'https://www.census.gov/naics/2022NAICS/2022_NAICS_Descriptions.xlsx'
    url_exclusions: str = 'https://www.census.gov/naics/2022NAICS/2022_NAICS_Cross_References.xlsx'

    sheet_codes: str = 'tbl_2022_title_description_coun'
    sheet_index: str = '2022NAICS'
    sheet_descriptions: str = '2022_NAICS_Descriptions'
    sheet_exclusions: str = '2022_NAICS_Cross_References'

    schema_codes: Dict[str, pl.DataType] = field(default_factory=lambda: {
        'Seq. No.': pl.UInt32,
        '2022 NAICS US   Code': pl.Utf8, 
        '2022 NAICS US Title': pl.Utf8
    }) # type: ignore
    schema_index: Dict[str, pl.DataType] = field(default_factory=lambda: {
        'NAICS22': pl.Utf8,
        'INDEX ITEM DESCRIPTION': pl.Utf8
    }) # type: ignore
    schema_descriptions: Dict[str, pl.DataType] = field(default_factory=lambda: {
        'Code': pl.Utf8,
        'Description': pl.Utf8
    }) # type: ignore
    schema_exclusions: Dict[str, pl.DataType] = field(default_factory=lambda: {
        'Code': pl.Utf8,
        'Cross-Reference': pl.Utf8
    }) # type: ignore
	
    rename_codes: Dict[str, str] = field(default_factory=lambda: {
        'Seq. No.': 'index',
        '2022 NAICS US   Code': 'code', 
        '2022 NAICS US Title': 'title'
    })        
    rename_index: Dict[str, str] = field(default_factory=lambda: {
        'NAICS22': 'code',
        'INDEX ITEM DESCRIPTION': 'examples_1'
    })
    rename_descriptions: Dict[str, str] = field(default_factory=lambda: {
        'Code': 'code',
        'Description': 'description'
    })
    rename_exclusions: Dict[str, str] = field(default_factory=lambda: {
        'Code': 'code',
        'Cross-Reference': 'excluded'
    })



# -------------------------------------------------------------------------------------------------
# 1. Import utility
# -------------------------------------------------------------------------------------------------

def read_naics_xlsx(
    url: str,
    sheet: str,
    schema: Dict[str, pl.DataType],
    cols: Dict[str, str]
) -> pl.DataFrame:

    '''
    Download and read a NAICS Excel file from Census Bureau.

    Args:
        url: URL to Excel file
        sheet: Sheet name to read
        schema: Column name to dtype mapping
        cols: Column rename mapping

    Returns:
        Polars DataFrame with renamed columns
    '''

    resp = httpx.get(url)
    resp.raise_for_status()
    data = resp.content

    f_xlsx = io.BytesIO(data)

    return (
        pl
        .read_excel(
            f_xlsx,
            sheet_name=sheet,
            columns=list(schema.keys()),
            schema_overrides=schema
        )
        .rename(mapping=cols)
    )


# -------------------------------------------------------------------------------------------------
# Read NAICS Titles
# -------------------------------------------------------------------------------------------------

cfg = Config()

# Load NAICS titles and normalize combined sector codes (31-33, 44-45, 48-49)
naics_titles = (
    read_naics_xlsx(
        url=cfg.url_codes,
        sheet=cfg.sheet_codes,
        schema=cfg.schema_codes,
        cols=cfg.rename_codes
    )
    .with_columns(
        code=pl.when(pl.col('code').eq('31-33')).then(pl.lit('31', pl.Utf8))
               .when(pl.col('code').eq('44-45')).then(pl.lit('44', pl.Utf8))
               .when(pl.col('code').eq('48-49')).then(pl.lit('48', pl.Utf8))
               .otherwise(pl.col('code'))
    )
    .select(
        index=pl.col('index')
                .sub(1),
        level=pl.col('code')
                .str.len_chars()
                .cast(pl.UInt8),
        code=pl.col('code'),

    )
)