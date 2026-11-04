import hashlib
import pytest
from experiments.r1.assets import checkpoint_record, dataset_record


def test_checkpoint_hash_does_not_load_and_rejects_pointer(tmp_path):
    file = tmp_path / 'test.pth'
    file.write_bytes(b'version https://git-lfs.github.com/spec/v1\noid sha256:abc\n')
    with pytest.raises(ValueError, match='LFS pointer'):
        checkpoint_record(file, 'https://example.org/reference')
    payload = b'synthetic fixture, not serialized weights'
    file.write_bytes(payload)
    record = checkpoint_record(file, 'https://example.org/reference')
    assert record['sha256'] == hashlib.sha256(payload).hexdigest()
    assert record['state_dict_validation'].startswith('NOT RUN')
    with pytest.raises(ValueError):
        checkpoint_record(file, '')


def test_split_preserves_order_and_requires_complete_triplets(tmp_path):
    (tmp_path / 'test.txt').write_text('b\na\n')
    with pytest.raises(FileNotFoundError):
        dataset_record(tmp_path, 'https://example.org/reference', expected_count=2)
    for folder in ('RGB', 'Modal', 'Label'):
        (tmp_path / folder).mkdir()
        for name in ('a', 'b'):
            (tmp_path / folder / (name + '.png')).write_bytes(b'synthetic hash fixture')
    record = dataset_record(tmp_path, 'https://example.org/reference', expected_count=2)
    assert record['files'][0]['path'] == 'RGB/b.png'
    assert record['sample_count'] == 2
    with pytest.raises(ValueError):
        dataset_record(tmp_path, 'https://example.org/reference')
    (tmp_path / 'test.txt').write_text('a\na\n')
    with pytest.raises(ValueError):
        dataset_record(tmp_path, 'https://example.org/reference', expected_count=2)
