#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MoleculeNet dataset catalog for molecular machine learning.

MoleculeNet is a benchmark for molecular machine learning containing
curated datasets spanning quantum mechanics, physical chemistry,
biophysics, and physiology.

Reference: https://moleculenet.org/
Data source: DeepChem's MoleculeNet catalog (static dataset list).
"""

from typing import Optional

from scitex_dev.decorators import supports_return_as

# MoleculeNet datasets are a curated benchmark suite; we fetch the catalog
# from DeepChem's GitHub repository which maintains the authoritative list.
MOLECULENET_API = "https://raw.githubusercontent.com/deepchem/deepchem/master/deepchem/molnet/load_function"

# Curated MoleculeNet benchmark catalog (from moleculenet.org)
_MOLECULENET_CATALOG = [
    {
        "id": "delaney",
        "name": "Delaney (ESOL)",
        "category": "physical_chemistry",
        "task_type": "regression",
        "n_compounds": 1128,
        "n_tasks": 1,
        "description": "Water solubility data (log solubility in mols per litre) for 1128 compounds.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1021/ci034243x",
    },
    {
        "id": "freesolv",
        "name": "FreeSolv",
        "category": "physical_chemistry",
        "task_type": "regression",
        "n_compounds": 642,
        "n_tasks": 1,
        "description": "Experimental and calculated hydration free energy of small molecules in water.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1007/s10822-014-9747-x",
    },
    {
        "id": "lipophilicity",
        "name": "Lipophilicity",
        "category": "physical_chemistry",
        "task_type": "regression",
        "n_compounds": 4200,
        "n_tasks": 1,
        "description": "Experimental results of octanol/water distribution coefficient (logD at pH 7.4).",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "",
    },
    {
        "id": "bace",
        "name": "BACE",
        "category": "biophysics",
        "task_type": "classification",
        "n_compounds": 1513,
        "n_tasks": 1,
        "description": "Binding results for inhibitors of human beta-secretase 1 (BACE-1).",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1021/acs.jcim.6b00290",
    },
    {
        "id": "bbbp",
        "name": "BBBP",
        "category": "biophysics",
        "task_type": "classification",
        "n_compounds": 2039,
        "n_tasks": 1,
        "description": "Blood-brain barrier penetration (permeability) dataset.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1021/ci300124c",
    },
    {
        "id": "clintox",
        "name": "ClinTox",
        "category": "physiology",
        "task_type": "classification",
        "n_compounds": 1478,
        "n_tasks": 2,
        "description": "Clinical trial toxicity and FDA approval status.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "",
    },
    {
        "id": "sider",
        "name": "SIDER",
        "category": "physiology",
        "task_type": "classification",
        "n_compounds": 1427,
        "n_tasks": 27,
        "description": "Side Effect Resource: drug side effects grouped into 27 system organ classes.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1093/nar/gkv1075",
    },
    {
        "id": "tox21",
        "name": "Tox21",
        "category": "physiology",
        "task_type": "classification",
        "n_compounds": 7831,
        "n_tasks": 12,
        "description": "Toxicology in the 21st Century: qualitative toxicity on 12 biological targets.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "",
    },
    {
        "id": "toxcast",
        "name": "ToxCast",
        "category": "physiology",
        "task_type": "classification",
        "n_compounds": 8576,
        "n_tasks": 617,
        "description": "EPA's ToxCast program: toxicology data for a large library of compounds.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "",
    },
    {
        "id": "muv",
        "name": "MUV",
        "category": "biophysics",
        "task_type": "classification",
        "n_compounds": 93087,
        "n_tasks": 17,
        "description": "Maximum Unbiased Validation datasets for virtual screening.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1021/ci8002649",
    },
    {
        "id": "hiv",
        "name": "HIV",
        "category": "biophysics",
        "task_type": "classification",
        "n_compounds": 41127,
        "n_tasks": 1,
        "description": "HIV replication inhibition screening results.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "",
    },
    {
        "id": "pcba",
        "name": "PCBA",
        "category": "biophysics",
        "task_type": "classification",
        "n_compounds": 437929,
        "n_tasks": 128,
        "description": "PubChem BioAssay data: biological activities of small molecules.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "",
    },
    {
        "id": "qm7",
        "name": "QM7",
        "category": "quantum_mechanics",
        "task_type": "regression",
        "n_compounds": 7165,
        "n_tasks": 1,
        "description": "Electronic properties (atomization energy) of small organic molecules.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1103/PhysRevLett.108.058301",
    },
    {
        "id": "qm8",
        "name": "QM8",
        "category": "quantum_mechanics",
        "task_type": "regression",
        "n_compounds": 21786,
        "n_tasks": 12,
        "description": "Electronic spectra and excited state energy of small molecules.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1063/1.4928757",
    },
    {
        "id": "qm9",
        "name": "QM9",
        "category": "quantum_mechanics",
        "task_type": "regression",
        "n_compounds": 133885,
        "n_tasks": 12,
        "description": "Geometric, energetic, electronic, and thermodynamic properties of DFT-modeled molecules.",
        "url": "https://moleculenet.org/datasets-1",
        "paper_doi": "10.1038/sdata.2014.22",
    },
]


@supports_return_as
def fetch_datasets() -> list[dict]:
    """
    Fetch the MoleculeNet dataset catalog.

    MoleculeNet is a static benchmark suite, so this returns the
    curated catalog rather than querying a live API.

    Returns
    -------
    list[dict]
        List of MoleculeNet dataset records.
    """
    return list(_MOLECULENET_CATALOG)


@supports_return_as
def fetch_all_datasets(
    max_datasets: Optional[int] = None,
    logger=None,
) -> list[dict]:
    """
    Fetch all MoleculeNet datasets.

    Parameters
    ----------
    max_datasets : int, optional
        Maximum number of datasets to return.
    logger : optional
        Logger for progress messages.

    Returns
    -------
    list[dict]
        List of MoleculeNet dataset records.
    """
    datasets = fetch_datasets()

    if logger:
        logger.info(f"MoleculeNet catalog: {len(datasets)} benchmark datasets")

    if max_datasets:
        datasets = datasets[:max_datasets]

    return datasets


@supports_return_as
def format_dataset(record: dict) -> dict:
    """
    Format a MoleculeNet dataset into a standardized dictionary.

    Parameters
    ----------
    record : dict
        Raw MoleculeNet dataset record.

    Returns
    -------
    dict
        Standardized dataset dictionary.
    """
    return {
        "id": record.get("id", ""),
        "doi": record.get("paper_doi", ""),
        "name": record.get("name", ""),
        "description": record.get("description", ""),
        "created": "",
        "modified": "",
        "version": "",
        "authors": [],
        "keywords": [record.get("category", ""), record.get("task_type", "")],
        "license": "MIT",
        "n_compounds": record.get("n_compounds", 0),
        "n_tasks": record.get("n_tasks", 0),
        "task_type": record.get("task_type", ""),
        "category": record.get("category", ""),
        "views": 0,
        "downloads": 0,
        "url": record.get("url", "https://moleculenet.org/"),
        "source": "moleculenet",
    }


# EOF
