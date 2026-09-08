# Example data

A small 2x2 grid cut from the China domain, stored ready to pass straight to
`embcca.correct` with no reshaping.

| File | Shape | Dtype | Axes |
|---|---|---|---|
| `test_grid_1992_2021_jja_model_DePreSys4.npy` | `(30, 100, 2, 2, 2)` | float32 | year, member, lon, lat, variable |
| `test_grid_1992_2021_jja_obs_ERA5_Land.npy` | `(30, 2, 2, 2)` | float64 | year, lon, lat, variable |

The variable axis is ordered **(precipitation, temperature)**: JJA total
precipitation then JJA mean temperature.

Years are 1992-2021, the analysis period used in the manuscript. The model's
100 members are the DePreSys4 realisation and leadtime axes already collapsed
together.

## Use

`examples/example.py` applies EMBCCA to these files, in both the gridded and
the area-mean form:

```bash
python examples/example.py
```

It needs only the package and NumPy.

## Note on the NetCDF files

The `China_*.nc` files here are the full-domain manuscript inputs and are not
committed. The data used in the paper is available on reasonable request; see
`paper/README.md`.
