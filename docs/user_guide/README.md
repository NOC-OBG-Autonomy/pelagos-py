Variable-processing pages, dropped in verbatim from the shared
"NOC Autonomy - Documentation" OneDrive folder. To update, overwrite the file
here with the new copy; to add one, copy it in and list it in
`docs/user_guide.rst`. Figures referenced as `../_static/<var>/...` live in
`docs/_static/`.

Figures: the PNGs the pages reference under `docs/_static/<var>/` are rendered
by the steps themselves (`BaseStep.diagnostic_figures`) from
`examples/configs/docs_chla_bbp.yaml`; the mapping of figure -> file lives in
`docs/scripts/figures.yaml`. Regenerate with `make figures` in `docs/` (needs
the docs dataset, see `docs/scripts/make_docs_dataset.py`); CI does the same.
