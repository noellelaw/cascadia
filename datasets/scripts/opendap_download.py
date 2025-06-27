import xarray as xr
import rioxarray
import os

# Parametri
anni = range(1901, 2000)  # date interval
base_url = "http://opendap.deltares.nl/thredds/dodsC/opendap/deltares/global_floods/EU-WATCH"

# Variabili da processare
dataset_info = {
    "pcrglob": {
        "variables": ["qw", "qloc"],  
    },
    "dynrout": {
        "variables": ["qc", "fldd"],  
    }
}

# new output dir
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)
for year in anni:
    date_str = f"{year}1231"
    for dataset_name, info in dataset_info.items():
        url = f"{base_url}/{dataset_name}/{date_str}_{dataset_name}.nc"

        try:
            ds = xr.open_dataset(url)
            n_days = ds.dims["time"]

            for day in range(n_days):
                for var in info["variables"]:
                    da = ds[var].isel(time=day)
                    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                    da.rio.write_crs("EPSG:4326", inplace=True)

                    full_path_out_dir = os.path.join(out_dir, dataset_name, var)
                    os.makedirs(full_path_out_dir, exist_ok=True)

                    out_name = f"{var}_{year}_day{day+1:03d}.tif"
                    out_path = os.path.join(out_dir, out_name)

                    if not os.path.exists(out_path):
                        da.rio.to_raster(out_path)
                        print(f"✅ Salvato: {out_path}")
                    else:
                        print(f"⏩ Già esistente: {out_path}")

        except Exception as e:
            print(f"❌ Error with {url}: {e}")