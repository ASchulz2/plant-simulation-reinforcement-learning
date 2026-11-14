from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TabularState:
    """
    Compact discrete state representation for tabular Q-learning.

    A ``TabularState`` contains the information used by the tabular
    Q-learning agent to distinguish routing situations. It represents
    the current material unit, the language composition of the
    AdmissionBuffer, and the language and occupancy of the two
    temporary storage areas.

    The order of material units inside the AdmissionBuffer is
    deliberately discarded. Only the number of units associated with
    each language is retained. Consequently, two AdmissionBuffer
    configurations with the same language counts produce the same
    tabular state, even when their internal MU ordering differs.

    The class is immutable and therefore suitable for use as a
    dictionary key in the sparse Q-table and the state registries.

    Attributes
    ----------
    current_language:
        Language of the material unit currently being considered by
        the routing decision.

    admission_buffer_ger_count:
        Number of German-language material units in the
        AdmissionBuffer.

    admission_buffer_fr_count:
        Number of French-language material units in the
        AdmissionBuffer.

    admission_buffer_rus_count:
        Number of Russian-language material units in the
        AdmissionBuffer.

    storage_a_language:
        Language currently associated with Storage A. An empty string
        represents an empty storage.

    storage_a_count:
        Current number of material units in Storage A.

    storage_b_language:
        Language currently associated with Storage B. An empty string
        represents an empty storage.

    storage_b_count:
        Current number of material units in Storage B.
    """

    current_language: str

    admission_buffer_ger_count: int
    admission_buffer_fr_count: int
    admission_buffer_rus_count: int

    storage_a_language: str
    storage_a_count: int

    storage_b_language: str
    storage_b_count: int
