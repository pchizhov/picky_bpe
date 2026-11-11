from .tokenization_picky_bpe import PickyBPETokenizer

__all__ = ['PickyBPETokenizer']


def _register_with_auto() -> None:
    """Make ``AutoTokenizer.from_pretrained(dir)`` find ``PickyBPETokenizer`` once this package is imported."""
    try:
        from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

        class PickyBPEConfig(PretrainedConfig):
            model_type = 'picky_bpe'

        AutoConfig.register('picky_bpe', PickyBPEConfig, exist_ok=True)
        AutoTokenizer.register(PickyBPEConfig, slow_tokenizer_class=PickyBPETokenizer, exist_ok=True)
    except Exception:  # registration is a convenience only
        pass


_register_with_auto()
