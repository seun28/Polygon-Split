import streamlit as st
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Polygon, MultiPolygon, box
import zipfile
import os
import tempfile
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_streamlit_page():
    """Configure the Streamlit page with title and instructions."""
    st.title("Shapefile Separator/Divider")
    st.write("""
    Please upload a shapefile (as a ZIP archive) and ensure the ZIP file contains all necessary shapefile components (.shp, .shx, .dbf, .prj).
    """)


def validate_zip_contents(zip_file):
    """
    Validate that the uploaded ZIP file contains required shapefile components.
    """
    try:
        file_list = zip_file.namelist()
        required_extensions = {".shp", ".shx", ".dbf"}
        optional_extensions = {".prj", ".cpg", ".qmd"}

        found_files = [Path(name) for name in file_list if not name.endswith("/")]
        found_extensions = {f.suffix.lower() for f in found_files}

        if not required_extensions.issubset(found_extensions):
            missing = required_extensions - found_extensions
            return False, f"Missing required files: {', '.join(missing)}"

        shp_files = [f for f in file_list if f.lower().endswith(".shp")]
        if len(shp_files) != 1:
            return False, "ZIP must contain exactly one .shp file"

        shp_path = Path(shp_files[0])
        parent_dir = str(shp_path.parent) + "/" if shp_path.parent != Path(".") else ""
        base_name = shp_path.stem

        for ext in required_extensions | optional_extensions:
            expected_file = f"{parent_dir}{base_name}{ext}"
            if ext in found_extensions and not any(
                f.lower() == expected_file.lower() for f in file_list
            ):
                return False, f"Mismatched filenames: {expected_file} not found"

        return True, ""
    except Exception as e:
        return False, f"Error validating ZIP contents: {str(e)}"


def subdivide_geometry(geometry, num_subdivisions):
    """
    Subdivide a geometry into approximately equal parts.
    """
    try:
        if geometry.is_empty:
            raise ValueError("Empty geometry encountered")

        minx, miny, maxx, maxy = geometry.bounds
        ratio = (maxx - minx) / (maxy - miny) if (maxy - miny) != 0 else 1
        cols = max(1, int(round((num_subdivisions * ratio) ** 0.5)))
        rows = max(1, int(round(num_subdivisions / cols)))

        width = (maxx - minx) / cols
        height = (maxy - miny) / rows

        subdivisions = []
        for i in range(rows):
            for j in range(cols):
                cell = box(
                    minx + j * width,
                    miny + i * height,
                    minx + (j + 1) * width,
                    miny + (i + 1) * height,
                )
                intersection = cell.intersection(geometry)
                if not intersection.is_empty and intersection.area > 0:
                    subdivisions.append(intersection)

        if not subdivisions:
            raise ValueError("No valid subdivisions created")

        return subdivisions
    except Exception as e:
        logger.error(f"Error subdividing geometry: {str(e)}")
        raise


def process_shapefile(input_gdf, num_subdivisions):
    """
    Process the input GeoDataFrame to create subdivisions.
    """
    try:
        new_geometries = []
        new_attributes = []
        total_features = len(input_gdf)
        progress_bar = st.progress(0)

        for idx, row in input_gdf.iterrows():
            progress = (idx + 1) / total_features
            progress_bar.progress(progress)

            if not row.geometry.is_valid:
                row.geometry = row.geometry.buffer(0)

            subdivided = subdivide_geometry(row.geometry, num_subdivisions)

            for sub_idx, sub_geom in enumerate(subdivided):
                new_geometries.append(sub_geom)
                attributes = row.drop("geometry").to_dict()
                attributes["subdivision_id"] = f"{idx}_{sub_idx}"
                new_attributes.append(attributes)

        return gpd.GeoDataFrame(
            new_attributes, geometry=new_geometries, crs=input_gdf.crs
        )
    except Exception as e:
        logger.error(f"Error processing shapefile: {str(e)}")
        raise


def save_gdf_to_shapefile(gdf, output_dir):
    """
    Save GeoDataFrame to shapefile format.
    """
    try:
        output_path = os.path.join(output_dir, "subdivided.shp")
        gdf.to_file(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error saving shapefile: {str(e)}")
        raise


def main():
    setup_streamlit_page()

    uploaded_file = st.file_uploader(
        "Upload your shapefile as a ZIP archive",
        type="zip",
        help="Upload a ZIP file containing all shapefile components (.shp, .shx, .dbf, .prj, .cpg, .qmd)",
    )

    if uploaded_file is not None:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_dir = os.path.join(temp_dir, "input")
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                with zipfile.ZipFile(uploaded_file) as zip_ref:
                    is_valid, error_msg = validate_zip_contents(zip_ref)
                    if not is_valid:
                        st.error(error_msg)
                        return

                    zip_ref.extractall(input_dir)

                shp_files = list(Path(input_dir).rglob("*.[sS][hH][pP]"))
                if not shp_files:
                    st.error("No .shp file found after extraction!")
                    return

                shp_file = shp_files[0]

                try:
                    gdf = gpd.read_file(shp_file, encoding="utf-8")
                except Exception:
                    try:
                        gdf = gpd.read_file(shp_file, encoding="latin1")
                    except Exception as e:
                        st.error(f"Failed to read shapefile: {str(e)}")
                        raise

                num_subdivisions = st.number_input(
                    "Number of subdivisions per geometry",
                    min_value=2,
                    max_value=100,
                    value=4,
                    help="Specify how many subdivisions to create for each geometry",
                )

                if st.button("Process Shapefile"):
                    with st.spinner("Processing..."):
                        result_gdf = process_shapefile(gdf, num_subdivisions)
                        save_gdf_to_shapefile(result_gdf, output_dir)

                        output_zip_path = os.path.join(temp_dir, "result.zip")
                        with zipfile.ZipFile(output_zip_path, "w") as zipf:
                            for file in Path(output_dir).glob("subdivided.*"):
                                zipf.write(file, file.name)

                        with open(output_zip_path, "rb") as f:
                            zip_data = f.read()

                        st.success(
                            "Processing complete! Click the button above to download your results."
                        )

                        st.download_button(
                            label="Download Processed Shapefile",
                            data=zip_data,
                            file_name="subdivided_shapefile.zip",
                            mime="application/zip",
                        )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error("Processing error", exc_info=True)


if __name__ == "__main__":
    main()
