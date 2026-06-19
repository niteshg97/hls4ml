from pathlib import Path
import pytest

from hls4ml.writer.catapult_writer import CatapultWriter


class DummyConfig:
    def __init__(self, output_dir, write_tar):
        self.output_dir = str(output_dir)
        self.writer_config = {'WriteTar': write_tar}

    def get_output_dir(self):
        return self.output_dir

    def get_writer_config(self):
        return self.writer_config


class DummyModel:
    def __init__(self, output_dir, write_tar):
        self.config = DummyConfig(output_dir, write_tar)


@pytest.mark.parametrize('write_tar', [True, False])
def test_catapult_write_tar_respects_writer_config(tmp_path, write_tar):
    output_dir = tmp_path / 'myproject'
    output_dir.mkdir()
    (output_dir / 'marker.txt').write_text('marker')

    CatapultWriter().write_tar(DummyModel(output_dir, write_tar))

    assert Path(f'{output_dir}.tar.gz').exists() == write_tar
