# SCOVA-CF v5 simultaneous-inference blocking report

The v5 simultaneous-inference evidence was re-aggregated successfully on
2026-07-19, but did not pass its statistical gates. It is archived as
non-promoted evidence and must not be used to authorize held-out validation.

- Protocol checksum: `7521cf977c51e97498ef7623c6facadfb8423a22e0740c2145d3ee7bbe68431b`
- Evidence checksum: `40f17a23a5cb4b46964ba5374e1ad54276158cbd669835ae0420a7474482f54b`
- Evidence run: `29706467376`

Four of six focused cells failed. Two weak-support, small-sample cells produced
typed small-library or empty-cell refusals; two other cells exceeded the frozen
familywise-error or family-coverage gates. The problem was the focused-cell
selection: it included settings outside the v5 candidate profile, which permits
only two or three groups with at least 50 observed units per group.

V6 is an inference-only amendment. It binds the v5 candidate and successful
external-agreement evidence by checksum, introduces a fresh inference seed
namespace, and evaluates only strong-support cells within the candidate profile.
V5 validation seeds remain untouched.
