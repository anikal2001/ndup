"""MinHash signatures over 5-word shingles."""

from __future__ import annotations

from typing import cast

import numpy as np
from datasketch import MinHash

from ndup.text.shingles import normalize


def minhash_signature(text: str, num_perm: int = 64) -> bytes:
    signer = MinHash(num_perm=num_perm)
    words = normalize(text).split()
    if len(words) < 5:
        if words:
            signer.update(" ".join(words).encode())
    else:
        for index in range(len(words) - 4):
            signer.update(" ".join(words[index : index + 5]).encode())
    return cast(bytes, signer.hashvalues.astype(np.uint64).tobytes())
