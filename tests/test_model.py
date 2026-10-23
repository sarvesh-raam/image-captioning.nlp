import torch

from export_model import main as export_main
from inference import CaptionGenerator, greedy_decode, load_model, sample_decode


def test_old_style_checkpoint_loads_and_decodes(tiny_checkpoint):
    model, vocab = load_model(tiny_checkpoint, torch.device("cpu"))
    memory = torch.randn(3, 49, 64)
    sos, eos = vocab.stoi["<SOS>"], vocab.stoi["<EOS>"]
    with torch.no_grad():
        assert greedy_decode(model, memory, sos, eos, max_len=5).shape[0] == 3
        assert sample_decode(model, memory, sos, eos, max_len=5).shape[0] == 3
    gen = CaptionGenerator(model, vocab, torch.device("cpu"))
    results = gen.beam_search(memory[:1], beam_size=3, max_len=5, num_results=3)
    assert results and len({tuple(ids) for ids, _ in results}) == len(results)


def test_head_only_loading_skips_backbone(tiny_checkpoint):
    model, _ = load_model(tiny_checkpoint, torch.device("cpu"), load_backbone=False)
    assert not hasattr(model.encoder, "resnet")


def test_export_halves_size(tiny_checkpoint, tmp_path, monkeypatch):
    out = tmp_path / "exported.pth"
    monkeypatch.setattr("sys.argv", ["export_model.py", tiny_checkpoint, str(out)])
    export_main()
    state = torch.load(out, weights_only=False)["model_state_dict"]
    assert all(v.dtype == torch.float16 for v in state.values() if v.is_floating_point())
    load_model(str(out), torch.device("cpu"))
