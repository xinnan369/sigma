"""Immutable receipt and per-run approvals for the frozen detector."""
from types import MappingProxyType
from typing import NamedTuple


class RunApproval(NamedTuple):
    receipt_signature_sha256: str
    runtime_signature_sha256: str


# PR #38 reviewed the original signatures. Human approval in PR #39 covers
# only these four runs. The runtime signature differs from the receipt
# signature solely in the fingerprint of the hardened authorization module.
TRUSTED_RECEIPTS = MappingProxyType({
    "eb002f9b3e4c82bdb6bbab6bb205f8ff5f88c0ddb13cc03b47d0796c9550effa": MappingProxyType({
        "A/3407": RunApproval(
            "9de4350b8eb555a86b6227085be2c37b9bcec1f412c617caaa668dbd5e58c72c",
            "d8690f32b4d5be1512f1c80afb0ae71bf76eb823f4398ef374d985d4efcdcdca",
        ),
        "B/3407": RunApproval(
            "3cb48ba7eba9c18227451f27815f957b2556c8ea759733b5b6ecf2cb83e0dccd",
            "f3ed70596cd5bf1ac4a2f9c821a0330d1c72172e12285759e80b6bcb15a9db02",
        ),
        "C/3407": RunApproval(
            "9a596f17cad224d7196226309a67e0f8036e8f11e9a187790b039607d7fd9dbc",
            "dbf70af2ebeb102d2d4c727ca06d2799fcb9400077ca024ba1b283faeb1fd18f",
        ),
        "D/3407": RunApproval(
            "a28b004765e54d3da99751e539650676e2f5eb00b76e68b90e385c108dde8473",
            "d0c030a7f5bd67b9060af48c099a431125cf5ede21e540b2e8cfa356063bb483",
        ),
    }),
})
