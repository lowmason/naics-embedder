# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import logging
import time
from typing import Optional

import httpx
import polars as pl

# -------------------------------------------------------------------------------------------------
# Download with exponential backoff retry
# -------------------------------------------------------------------------------------------------

def download_with_retry(
    url: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    timeout: float = 30.0
) -> Optional[bytes]:
    
    '''
    Download content from URL with exponential backoff retry logic.
    
    Returns:
        bytes: The downloaded content
        
    Raises:
        httpx.HTTPError, httpx.TimeoutException, ValueError: If all retries fail
    '''

    last_exception = None
    
    for attempt in range(max_retries + 1):

        try:
            
            resp = httpx.get(url, timeout=timeout)
            resp.raise_for_status()
            data = resp.content
            
            if not data:
                raise ValueError(f'Empty response received from {url}')
            
            return data
            
        except (httpx.HTTPError, httpx.TimeoutException, ValueError) as e:

            last_exception = e
            
            if attempt < max_retries:

                delay = initial_delay * (backoff_factor ** attempt)

                print(f'Attempt {attempt + 1}/{max_retries + 1} failed for {url}: {str(e)}')
                print(f'Retrying in {delay:.1f} seconds...')
                
                time.sleep(delay)

            else:

                print(f'All {max_retries + 1} attempts failed for {url}')

                raise last_exception


# -------------------------------------------------------------------------------------------------
# Parquet stats
# -------------------------------------------------------------------------------------------------

def parquet_stats(
    parquet_df: pl.DataFrame,
    message: str,
    output_parquet: str,
    logger: logging.Logger
) -> None:

    logger.info(f'\nParquet observations: {parquet_df.height: ,}\n')

    schema = (
        parquet_df
        .schema
    )

    rows = [(n, d) for n, d in zip(schema.names(), schema.dtypes())]

    logger.info('Parquet schema: Schema([')
    for name, dtype in rows:
        logger.info(f"    ('{name}', {dtype}),")
    logger.info('])\n')

    logger.info(f'{parquet_df.height: ,} {message}:')
    logger.info(f'  {output_parquet}\n')