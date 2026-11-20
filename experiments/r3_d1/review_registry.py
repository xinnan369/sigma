"""Immutable receipt and per-run approvals for the frozen detector."""
from types import MappingProxyType
from typing import NamedTuple


class RunApproval(NamedTuple):
    receipt_signature_sha256: str
    runtime_signature_sha256: str


# PR #38 reviewed the original signatures. PR #39 records approval for 3407.
# The explicit R7 scope extension approves the other sixteen runs.
# The runtime signature differs from the receipt
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
        "A/3408": RunApproval(
            "57c1f6ea8567d2600c8d787420c5e20712ba4d15fd17538b7ea4b2fcd33461a9",
            "07b079dc329bc99b7d00b53bcea33ac2f57a8020b5e0401c94f0e97690ec0f1d",
        ),
        "B/3408": RunApproval(
            "f80dc04a857bb406aa177c71655fa24c11d337273b9fe76db1490894e81576bf",
            "241540b30de42a6083ba8dfa934d0e34cd4ac8c01017588d2e9f79bb78c7c536",
        ),
        "C/3408": RunApproval(
            "d4fa4cdf99e4871073df42641adac9a56182a31d596302238647422ae767c022",
            "c07464daca819d6f5641e4d5656e29e273cfacde28e719e9589b661b439ef3fc",
        ),
        "D/3408": RunApproval(
            "5af8f709dc899389ca26c878b33a7180f87677c60b598b7e4dce7ef0b650317d",
            "bc7fc8e221de9586a53cbb168092c89872684f568b10c75a863f6b95eb7dd7e1",
        ),
        "A/3409": RunApproval(
            "982f0cafd566dccdafa75c55f57384f083bd278540e70adf3bae309121721cd3",
            "a6d0c427eb965059c4f74f861e4641fe235de669bc84c3cfc7363b15fcf81958",
        ),
        "B/3409": RunApproval(
            "0edeb43b0f747aa6bdcd5767b8abfa55ce7ca07f6b771db0bbfad60fa084e27e",
            "ace90dd2cfd17682e634c43605aa42f4d15a9b62e7026129fa754e205e46b923",
        ),
        "C/3409": RunApproval(
            "8edf4b883ced9967c1b9e05943a7cba4fa2470e79e912b38f7d50ac138be2bcf",
            "755bdb82f7d470f6821a6fdecda3ff6ccd80e39df4657fb65f85323c15420362",
        ),
        "D/3409": RunApproval(
            "fa6c377a22b82e8bb59d8fcb80a47d441648d266f7aacfa13c3d9e6fbf94ac7a",
            "e4f09ce97e25d7509e4e527cc05d46e6106d6343a976a1be5a4cbd83e81b9451",
        ),
        "A/3410": RunApproval(
            "978bdd982268d953e58458302d7f3872cd8e45a99e7c36c2f86fdd56615baba6",
            "b4ff9b6011526f85b027c1cd991bc3ffc6f3384377220dccc0285db7addeae3c",
        ),
        "B/3410": RunApproval(
            "70fb31641d054d85ba6c613729ec65c7764704bf6758efd2b11658f24882a1a5",
            "08ef1702f45721349510c95db212ccc44f873c9d1f0345a2d00034460ffdd99a",
        ),
        "C/3410": RunApproval(
            "dba9e6fe7e974829938eaec0dfd7d143ccd454250d023da6c528d339c5119b49",
            "fb47d2514e0af4c5c44c94f185d62923504dfee8f6801def6c6e62023773d5d0",
        ),
        "D/3410": RunApproval(
            "a1475ab6a4d144300cfd0e0cc1af1aedd5f6334431a2c22e2c94a75a1ee0ce3c",
            "e30527262a4cbe6c2ca73e0e8f98377734ded1f1cc4f622e388dd177f9515367",
        ),
        "A/3411": RunApproval(
            "a715b45b9743a57fd24aa768f80684d8970ecafee6aa4b14e2a329c9118fbe49",
            "88c3cc1fc16e4c3db012ebd5b43a0ed358490dc98bb835f28f3159bec54de3d4",
        ),
        "B/3411": RunApproval(
            "5e7079197e95060424e11881bf135474e13ddc76c870fac47632cf25a1b46650",
            "f5b86025d044183b0a1c1823e83244f5cb5328b444b7ae8f6800fce7b89c032b",
        ),
        "C/3411": RunApproval(
            "2216bdd365967802480a0492b09776954378b40e1f996b5bec6ac9ba1fce28aa",
            "abc53372c1d47738a696c74af7ffda61c751072b4ff6674e0a39fc588bbd79e0",
        ),
        "D/3411": RunApproval(
            "c2850bcdda6c9e805e68478e97039d9e540be09779d9807b023214d999194425",
            "5d5d9fab04b887420b6027ac86158b80b1fc2dd463820cb7cb6af11f99db1048",
        ),
    }),
})
