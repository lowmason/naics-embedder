'''Plot the durable epoch health and monitor values; no evaluation split is read (P20).'''

import pytest

from naics_embedder.tools._visualize_metrics import HAS_MATPLOTLIB, create_visualizations
from tests.fixtures.epoch_summary import summary_rows

pytestmark = pytest.mark.unit

class TestCreateVisualizations:

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
    def test_create_visualizations_creates_output_file(self, tmp_path):
        path = create_visualizations(summary_rows(), tmp_path)
        assert path == tmp_path / 'epoch_metrics.png'
        assert path.exists() and path.stat().st_size > 0

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
    def test_create_visualizations_creates_output_dir(self, tmp_path):
        directory = tmp_path / 'nested' / 'output'
        create_visualizations(summary_rows(), directory)
        assert directory.exists()

    def test_create_visualizations_handles_empty_metrics(self, tmp_path):
        with pytest.raises(ValueError, match='No metrics'):
            create_visualizations([], tmp_path)
        assert not list(tmp_path.iterdir())

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
    def test_create_visualizations_handles_single_epoch(self, tmp_path):
        path = create_visualizations(summary_rows()[:1], tmp_path)
        assert path.exists() and path.stat().st_size > 0

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
def test_plots_use_every_durable_field_including_moe_and_radius_sd(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    plots, bands = [], []
    original_plot, original_fill = Axes.plot, Axes.fill_between

    def plot_spy(self, x, y, *args, **kwargs):
        plots.append((kwargs.get('label'), list(x), list(y)))
        return original_plot(self, x, y, *args, **kwargs)

    def band_spy(self, x, low, high, *args, **kwargs):
        bands.append((list(x), list(low), list(high)))
        return original_fill(self, x, low, high, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', plot_spy)
    monkeypatch.setattr(Axes, 'fill_between', band_spy)
    rows = summary_rows()
    for row in rows:
        row['loss/load_balancing'] = 0.2 + row['epoch'] * 0.1
    create_visualizations(rows, tmp_path)
    curves = {label: (x, y) for label, x, y in plots}
    assert set(curves) == {
        'mrr', 'loss/task', 'loss/code_code', 'loss/radial', 'loss/total', 'loss/load_balancing',
        'logit_scale/task', 'logit_scale/code_code', 'radius/mean/level_2', 'radius/mean/level_6'
    }
    for key, (x, y) in curves.items():
        assert x == [0, 1, 2]
        assert y == [row[key] for row in rows]
    assert len(bands) == 2
    for (x, low, high), level in zip(bands, [2, 6]):
        assert x == [0, 1, 2]
        assert low == pytest.approx(
            [row[f'radius/mean/level_{level}'] - row[f'radius/sd/level_{level}'] for row in rows]
        )
        assert high == pytest.approx(
            [row[f'radius/mean/level_{level}'] + row[f'radius/sd/level_{level}'] for row in rows]
        )

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
def test_missing_samples_are_omitted_instead_of_plotted_as_zero(tmp_path, monkeypatch):
    from matplotlib.axes import Axes
    curves, bands = {}, []
    original_plot, original_fill = Axes.plot, Axes.fill_between

    def plot_spy(self, x, y, *args, **kwargs):
        curves[kwargs.get('label')] = (list(x), list(y))
        return original_plot(self, x, y, *args, **kwargs)

    def band_spy(self, x, low, high, *args, **kwargs):
        bands.append((list(x), list(low), list(high)))
        return original_fill(self, x, low, high, *args, **kwargs)

    monkeypatch.setattr(Axes, 'plot', plot_spy)
    monkeypatch.setattr(Axes, 'fill_between', band_spy)
    rows = summary_rows()
    del rows[1]['loss/task']
    del rows[1]['radius/sd/level_2']
    rows[1]['mrr'] = None
    create_visualizations(rows, tmp_path)
    assert curves['loss/task'] == ([0, 2], [rows[0]['loss/task'], rows[2]['loss/task']])
    assert curves['mrr'] == ([0, 2], [0.5, 0.7])
    assert curves['radius/mean/level_2'] == (
        [0, 1, 2], [row['radius/mean/level_2'] for row in rows]
    )
    assert len(bands) == 2
    x, low, high = bands[0]
    assert x == [0, 2]
    assert low == pytest.approx(
        [rows[i]['radius/mean/level_2'] - rows[i]['radius/sd/level_2'] for i in [0, 2]]
    )
    assert high == pytest.approx(
        [rows[i]['radius/mean/level_2'] + rows[i]['radius/sd/level_2'] for i in [0, 2]]
    )

@pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib not available')
def test_radius_band_matches_its_mean_color_when_an_earlier_level_has_no_sd(tmp_path, monkeypatch):
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure

    figures = []
    original_savefig = Figure.savefig

    def savefig_spy(self, *args, **kwargs):
        figures.append(self)
        return original_savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', savefig_spy)
    rows = summary_rows()
    for row in rows:
        del row['radius/sd/level_2']
    create_visualizations(rows, tmp_path)

    radius_axis = figures[0].axes[3]
    curves = {line.get_label(): line for line in radius_axis.lines}
    assert set(curves) == {'radius/mean/level_2', 'radius/mean/level_6'}
    assert len(radius_axis.collections) == 1
    level_2_color = to_rgba(curves['radius/mean/level_2'].get_color())
    level_6_color = to_rgba(curves['radius/mean/level_6'].get_color())
    band_color = radius_axis.collections[0].get_facecolor()[0]
    assert level_2_color != level_6_color
    assert band_color == pytest.approx((*level_6_color[:3], 0.2))
