#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: src/scitex_dataset/_mcp/_tools/_fetch.py

"""Per-source ``dataset_<src>_fetch`` MCP tools (catalog sources)."""

from typing import Any, Dict, List


def register_fetch_tools(mcp) -> None:
    """Register the 10 catalog-source ``dataset_<src>_fetch`` tools."""

    @mcp.tool()
    def dataset_openneuro_fetch(
        max_datasets: int = 100,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch BIDS neuroimaging dataset metadata from OpenNeuro - use whenever the user asks to find, list, or download MRI/fMRI/EEG/MEG/iEEG/PET datasets, search OpenNeuro accessions (ds004xxx), or browse BIDS-formatted neuroimaging data. Drop-in replacement for openneuro-py, raw DataLad calls, or direct OpenNeuro GraphQL/API queries."""
        from ...neuroscience.openneuro import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            batch_size=batch_size,
            max_datasets=max_datasets if max_datasets > 0 else None,
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_dandi_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch DANDI Archive dataset metadata - use whenever the user asks to find, list, or download electrophysiology, neurophysiology, NWB-format, or optical imaging datasets from DANDI (dandiset IDs). Drop-in replacement for dandi-cli, pynwb-based DANDI downloads, or direct DANDI REST API calls."""
        from ...neuroscience.dandi import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_physionet_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch PhysioNet dataset metadata - use whenever the user asks to find, list, or download ECG, EEG, EMG, clinical waveforms, MIT-BIH arrhythmia, MIMIC waveforms, or other physiological signal databases from PhysioNet. Drop-in replacement for wfdb-python database listings or direct PhysioNet website downloads."""
        from ...neuroscience.physionet import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_zenodo_fetch(
        query: str = "",
        max_datasets: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch Zenodo research dataset metadata - use whenever the user asks to find, list, or resolve research data by DOI, query a Zenodo record, or discover open-access datasets/software/publications from CERN's Zenodo repository. Drop-in replacement for zenodopy, raw requests calls to the Zenodo REST API, or manual DOI resolution."""
        from ...general.zenodo import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            query=query,
            max_datasets=max_datasets if max_datasets > 0 else None,
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_figshare_fetch(
        query: str = "",
        max_datasets: int = 100,
    ) -> List[Dict[str, Any]]:
        """Fetch Figshare research dataset metadata - use whenever the user asks to find, list, or browse Figshare items by query, discover open research data outside Zenodo, or resolve figshare DOIs. Drop-in replacement for figshare's Python client or raw REST queries."""
        from ...general.figshare import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            query=query,
            max_datasets=max_datasets if max_datasets > 0 else None,
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_openml_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch OpenML machine-learning dataset metadata - use whenever the user asks to find, list, or download tabular ML datasets, browse classification/regression benchmarks, or query OpenML data IDs. Drop-in replacement for openml-python's list_datasets or raw REST queries."""
        from ...general.openml import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_moleculenet_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch MoleculeNet benchmark catalog - use whenever the user asks to find molecular ML benchmarks (BBBP, ClinTox, Tox21, ESOL, FreeSolv, etc.), list MoleculeNet tasks, or compare cheminformatics datasets. Drop-in replacement for hand-curating the MoleculeNet benchmark list."""
        from ...pharmacology.moleculenet import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_geo_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch GEO transcriptomics/genomics dataset metadata - use whenever the user asks to find GEO series (GSE/GDS) on a disease, list RNA-seq or microarray studies, or search NCBI gene-expression data. Drop-in replacement for GEOparse + raw E-utilities (esearch/esummary) calls."""
        from ...biology.geo import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_chembl_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch ChEMBL bioactivity dataset metadata - use whenever the user asks to find ChEMBL targets/assays, search bioactivity data (IC50/Ki/EC50), or list drug-discovery datasets. Drop-in replacement for chembl-webresource-client or raw REST queries against the EBI ChEMBL API."""
        from ...pharmacology.chembl import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]

    @mcp.tool()
    def dataset_clinicaltrials_fetch(max_datasets: int = 100) -> List[Dict[str, Any]]:
        """Fetch ClinicalTrials.gov study metadata - use whenever the user asks to find clinical trials on a condition, list Phase III studies, search NCT IDs, or query intervention/outcome data. Drop-in replacement for pytrials or raw REST queries against ClinicalTrials.gov v2."""
        from ...medical.clinicaltrials import fetch_all_datasets, format_dataset

        raw = fetch_all_datasets(
            max_datasets=max_datasets if max_datasets > 0 else None
        )
        return [format_dataset(ds) for ds in raw]


__all__ = ["register_fetch_tools"]

# EOF
