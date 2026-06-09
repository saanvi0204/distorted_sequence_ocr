class Vocabulary:
    """
    Maps characters to integer indices for CTC training and decoding.

    Index 0 is reserved as the CTC blank token — it acts as a separator
    between characters and is removed during decoding.  All other characters
    are assigned indices starting from 1, derived from the sorted set of
    characters found in the training labels.
    """

    BLANK_IDX = 0

    def __init__(self, labels: list[str]) -> None:
        chars = sorted(set("".join(labels)))
        self.char2idx = {char: idx + 1 for idx, char in enumerate(chars)}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    @property
    def vocab_size(self) -> int:
        """Total number of classes including the blank token."""
        return len(self.char2idx) + 1

    def encode(self, text: str) -> list[int]:
        """Convert a label string to a list of integer indices."""
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices: list[int], remove_duplicates: bool = True) -> str:
        """
        Convert a sequence of integer indices back to a string.

        Args:
            indices:           Raw argmax output from the model (one index per timestep).
            remove_duplicates: If True, apply the CTC merge rule — collapse consecutive
                               identical indices before removing blanks.
        """
        if remove_duplicates:
            indices = [v for i, v in enumerate(indices)
                       if i == 0 or v != indices[i - 1]]
        return "".join(
            self.idx2char[i]
            for i in indices
            if i != self.BLANK_IDX and i in self.idx2char
        )
