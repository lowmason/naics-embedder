'''
The regressor panel's branch record matches Stage 1's finding (roadmap Stage 3 Exit).

The record in ``conf/data/regressor_panel.yaml`` is compared field by field with the finding's
decision block, which Stage 3 reads verbatim, and with section 4's table of excluded codes.
'''

import re
from pathlib import Path

import pytest

from naics_embedder.utils.config import RegressorPanelConfig, load_config

pytestmark = pytest.mark.unit

FINDING = Path('specs/findings/employment-statistics-coverage.md')

@pytest.fixture(scope='module')
def finding() -> str:
    return FINDING.read_text(encoding='utf-8')

@pytest.fixture(scope='module')
def block(finding) -> str:
    match = re.search(r'<!-- decision:begin -->\n(.*?)<!-- decision:end -->', finding, re.DOTALL)
    assert match, 'the finding has no decision block'
    return match.group(1)

@pytest.fixture(scope='module')
def record():
    cfg = load_config(RegressorPanelConfig, 'data/regressor_panel.yaml')
    assert cfg.branch_record is not None
    return cfg.branch_record

def _field(block: str, name: str) -> str:
    match = re.search(rf'^- \*\*{re.escape(name)}:\*\* (.+)$', block, re.MULTILINE)
    assert match, f'the decision block has no {name} line'
    return match.group(1)

def _count(text: str) -> int:
    return int(text.replace(',', ''))

def test_the_branch_and_both_answers_match_the_decision_block(record, block):
    assert _field(block, 'Branch').startswith(f'{record.branch}. ')
    assert _field(block, 'Time-respecting outcome') == (
        'yes.' if record.time_respecting_outcome else 'no.'
    )
    assert _field(block, 'Seen-code regime') == ('yes.' if record.seen_regime else 'no.')

def test_the_source_years_ownership_and_grain_match_the_decision_block(record, block):
    source = _field(block, 'Source')

    assert source.startswith(f'{record.source}, ')
    years = re.search(r'reference years ((?:\d{4}, )*\d{4})', source).group(1)
    assert [int(year) for year in years.split(', ')] == record.reference_years
    assert re.search(r'\(own_code (\d+)\)', source).group(1) == record.ownership
    assert f'({record.grain}, private ownership)' in _field(block, 'Row grain')

def test_the_population_matches_the_decision_block(record, block):
    seen, heldout, codebook = re.search(
        r'([\d,]+) codes for the seen-code regime and ([\d,]+) for the held-out-code regime, '
        r'of the ([\d,]+) six-digit codes',
        _field(block, 'Population'),
    ).groups()

    assert (_count(seen), _count(heldout)) == (record.population_seen, record.population_heldout)
    assert _count(codebook) - len(record.excluded_codes) == record.population_seen

def test_the_excluded_codes_are_section_fours(record, finding):
    section = finding.split('### Codes excluded at the chosen grain', 1)[1].split('\n## ', 1)[0]

    excluded = re.findall(r'^\| (\d{6}) \| no private cell \|$', section, re.MULTILINE)

    assert len(excluded) == 32
    assert record.excluded_codes == excluded
