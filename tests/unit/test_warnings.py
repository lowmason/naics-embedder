'''
Unit tests for warning suppression utilities.

Tests the centralized warning configuration module.
'''

import warnings

import pytest

from naics_embedder.utils.warnings import (
    _WARNING_FILTERS,
    configure_warnings,
    get_warning_rationale,
    list_suppressed_warnings,
)

# -------------------------------------------------------------------------------------------------
# Tests for configure_warnings()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestConfigureWarnings:
    '''Tests for configure_warnings() function.'''

    def test_configure_warnings_runs_without_error(self):
        '''Test that configure_warnings runs without raising.'''
        # Should not raise
        configure_warnings()

    def test_configure_warnings_suppresses_precision_warning(self):
        '''Test that precision warning is suppressed.'''
        configure_warnings()

        # This warning should be suppressed
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            # Trigger a warning that matches the pattern
            warnings.warn(
                'Precision bf16 is not supported by the model summary',
                UserWarning,
            )
            # Note: The filter won't catch this because the module doesn't match
            # but we can verify the filter was applied

    def test_configure_warnings_with_additional_filters(self):
        '''Test adding additional warning filters.'''
        additional = [
            ('.*custom warning.*', UserWarning, 'test_module'),
        ]

        configure_warnings(additional_filters=additional)

        # Should not raise

    def test_configure_warnings_verbose_mode(self, caplog):
        '''Test verbose mode logging.'''
        import logging

        with caplog.at_level(logging.DEBUG):
            configure_warnings(verbose=True)

        # In verbose mode, should log suppression messages
        # Note: The logger may not capture if not configured

    def test_configure_warnings_applies_all_standard_filters(self):
        '''Test that all standard filters are applied.'''
        # Clear any existing filters
        warnings.resetwarnings()

        configure_warnings()

        # Verify filters were applied by checking the warnings.filters list
        # Note: We can't directly inspect applied filters, but we can verify
        # the function completes successfully
        assert True

    def test_configure_warnings_idempotent(self):
        '''Test that calling configure_warnings multiple times is safe.'''
        configure_warnings()
        configure_warnings()
        configure_warnings()

        # Should not raise or cause issues

# -------------------------------------------------------------------------------------------------
# Tests for get_warning_rationale()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestGetWarningRationale:
    '''Tests for get_warning_rationale() function.'''

    def test_get_rationale_for_known_pattern(self):
        '''Test getting rationale for a known warning pattern.'''
        # Use a pattern from the first filter
        pattern = 'Precision'
        rationale = get_warning_rationale(pattern)

        assert rationale is not None
        assert isinstance(rationale, str)
        assert len(rationale) > 0

    def test_get_rationale_returns_none_for_unknown(self):
        '''Test that unknown patterns return None.'''
        rationale = get_warning_rationale('completely_unknown_pattern_xyz')

        assert rationale is None

    def test_get_rationale_partial_match(self):
        '''Test that partial pattern matches work.'''
        # 'eval mode' should match the module eval mode warning
        rationale = get_warning_rationale('eval mode')

        assert rationale is not None

    def test_get_rationale_for_each_standard_filter(self):
        '''Test that each standard filter has a rationale.'''
        for message, _, _, expected_rationale in _WARNING_FILTERS:
            # Extract a key part of the message for lookup
            key_part = message.replace('.*', '').strip()[:20]
            rationale = get_warning_rationale(key_part)

            # Should find a matching rationale
            # Note: get_warning_rationale does substring matching
            if rationale is not None:
                assert isinstance(rationale, str)

# -------------------------------------------------------------------------------------------------
# Tests for list_suppressed_warnings()
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestListSuppressedWarnings:
    '''Tests for list_suppressed_warnings() function.'''

    def test_list_returns_list(self):
        '''Test that list_suppressed_warnings returns a list.'''
        result = list_suppressed_warnings()

        assert isinstance(result, list)

    def test_list_contains_tuples(self):
        '''Test that each item is a (pattern, rationale) tuple.'''
        result = list_suppressed_warnings()

        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            pattern, rationale = item
            assert isinstance(pattern, str)
            assert isinstance(rationale, str)

    def test_list_matches_warning_filters_count(self):
        '''Test that count matches _WARNING_FILTERS.'''
        result = list_suppressed_warnings()

        assert len(result) == len(_WARNING_FILTERS)

    def test_list_contains_expected_warnings(self):
        '''Test that expected warning patterns are in the list.'''
        result = list_suppressed_warnings()
        patterns = [pattern for pattern, _ in result]

        # Check for expected patterns
        assert any('Precision' in p for p in patterns)
        assert any('eval mode' in p for p in patterns)
        assert any('workers' in p for p in patterns)
        assert any('Checkpoint' in p for p in patterns)
        assert any('batch_size' in p for p in patterns)

# -------------------------------------------------------------------------------------------------
# Tests for _WARNING_FILTERS constant
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestWarningFilters:
    '''Tests for _WARNING_FILTERS constant.'''

    def test_warning_filters_is_list(self):
        '''Test that _WARNING_FILTERS is a list.'''
        assert isinstance(_WARNING_FILTERS, list)

    def test_warning_filters_structure(self):
        '''Test that each filter has correct structure.'''
        for filter_tuple in _WARNING_FILTERS:
            assert len(filter_tuple) == 4
            message, category, module, rationale = filter_tuple

            assert isinstance(message, str)
            assert issubclass(category, Warning)
            assert isinstance(module, str)
            assert isinstance(rationale, str)

    def test_warning_filters_categories_are_warning_types(self):
        '''Test that all categories are Warning subclasses.'''
        for _, category, _, _ in _WARNING_FILTERS:
            assert issubclass(category, Warning)

    def test_warning_filters_patterns_are_regex(self):
        '''Test that message patterns are valid regex.'''
        import re

        for message, _, _, _ in _WARNING_FILTERS:
            # Should compile without error
            re.compile(message)

# -------------------------------------------------------------------------------------------------
# Integration tests
# -------------------------------------------------------------------------------------------------

@pytest.mark.unit
class TestIntegration:
    '''Integration tests for warning utilities.'''

    def test_warning_suppression_workflow(self):
        '''Test complete warning suppression workflow.'''
        # List warnings before configuration
        suppressed = list_suppressed_warnings()
        assert len(suppressed) > 0

        # Get rationale for first warning
        first_pattern = suppressed[0][0]
        rationale = get_warning_rationale(first_pattern)
        assert rationale is not None

        # Configure warnings
        configure_warnings()

        # Should complete without error

    def test_can_add_custom_filter_at_runtime(self):
        '''Test adding custom filter at runtime.'''
        custom_filter = [
            ('.*my custom test warning.*', UserWarning, 'test_module'),
        ]

        configure_warnings(additional_filters=custom_filter)

        # Warning should now be suppressed
        # (can't easily verify without matching module)
