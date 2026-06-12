import pandas
from matplotlib import pyplot

from physics_applications_of_ai.artifacts import save_figure, save_table


def test_save_table_writes_csv(tmp_path):
    table = pandas.DataFrame({"metric": [1.0]})

    output_path = save_table(table, tmp_path, "metrics.csv")

    assert output_path.exists()
    assert "metric" in output_path.read_text()


def test_save_figure_writes_png(tmp_path):
    figure, axis = pyplot.subplots()
    axis.plot([0, 1], [0, 1])

    output_path = save_figure(figure, tmp_path, "plot.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    pyplot.close(figure)
