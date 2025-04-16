# Import required libraries for web interface
import streamlit as st
# Import library for handling geospatial data
import geopandas as gpd
# Import library for numerical operations
import numpy as np
# Import library for data manipulation and analysis
import pandas as pd
# Import library for geometric operations
import shapely
# Import specific geometric types from shapely
from shapely.geometry import Polygon, MultiPolygon, box, LineString, Point
# Import geometric operations for splitting polygons
from shapely.ops import split, unary_union
# Import library for handling ZIP files
import zipfile
# Import library for operating system operations
import os
# Import library for creating temporary directories
import tempfile
# Import library for handling file paths
from pathlib import Path
# Import library for logging
import logging
# Import type hints for better code documentation
from typing import List, Union, Tuple, Optional

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO)
# Create a logger instance for this module
logger = logging.getLogger(__name__)


def setup_streamlit_page():
    """Configure the Streamlit page with title and instructions."""
    st.set_page_config(layout="wide")
    st.title("Polygon Equal Area Subdivider")
    st.write("""
    Upload a shapefile (as a ZIP archive or .shp file) containing polygons to subdivide into equal areas.
    The ZIP file should contain all necessary shapefile components (.shp, .shx, .dbf, .prj).
    Each polygon will be subdivided into equal areas based on the number of subdivisions specified.
    """)


def validate_zip_contents(zip_file) -> Tuple[bool, str]:
    """
    Validate that the uploaded ZIP file contains required shapefile components.
    Returns (is_valid, message_or_basename).
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

        return True, base_name
    except Exception as e:
        return False, f"Error validating ZIP contents: {str(e)}"


def project_point(point: Point, angle_deg: float) -> float:
    """Project a point onto a direction given by angle_deg (in degrees)."""
    angle_rad = np.radians(angle_deg)
    x, y = point.x, point.y
    return x * np.cos(angle_rad) + y * np.sin(angle_rad)


def get_projection_bounds(polygon: Union[Polygon, MultiPolygon], angle_deg: float) -> Tuple[float, float]:
    """Get the min and max projections of the polygon's vertices along the given angle."""
    if isinstance(polygon, MultiPolygon):
        polygon = next(iter(polygon.geoms))
    
    vertices = []
    exterior = polygon.exterior
    vertices.extend([Point(x, y) for x, y in exterior.coords])
    for interior in polygon.interiors:
        vertices.extend([Point(x, y) for x, y in interior.coords])
    
    projections = [project_point(pt, angle_deg) for pt in vertices]
    return min(projections), max(projections)


def analyze_polygon_shape(polygon: Union[Polygon, MultiPolygon]) -> float:
    """Analyze polygon shape to determine best split angle using minimum rotated rectangle."""
    mrr = polygon.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    dx = coords[1][0] - coords[0][0]
    dy = coords[1][1] - coords[0][1]
    angle_rad = np.arctan2(dy, dx)
    split_angle = (np.degrees(angle_rad) + 90) % 180
    return split_angle


def create_split_line(polygon: Union[Polygon, MultiPolygon], split_angle: float, offset: float) -> LineString:
    """Create a split line at given angle and offset along the perpendicular direction (split_angle + 90)."""
    projection_angle = split_angle + 90
    theta_rad = np.radians(projection_angle)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    minx, miny, maxx, maxy = polygon.bounds
    width = maxx - minx
    height = maxy - miny
    diagonal = np.hypot(width, height)
    
    if sin_theta != 0:
        x1 = 0
        y1 = (offset - x1 * cos_theta) / sin_theta
    else:
        x1 = offset / cos_theta if cos_theta != 0 else 0
        y1 = 0
    
    x2 = x1 + diagonal * np.cos(np.radians(split_angle))
    y2 = y1 + diagonal * np.sin(np.radians(split_angle))
    
    if sin_theta == 0:
        y1 = miny - diagonal
        y2 = maxy + diagonal
    
    return LineString([(x1, y1), (x2, y2)])


def split_polygon_equally(geometry: Union[Polygon, MultiPolygon], n_parts: int) -> List[Union[Polygon, MultiPolygon]]:
    """
    Split a polygon into n equal parts using an intelligent shape-based approach.
    """
    if n_parts == 1:
        return [geometry]

    if not isinstance(geometry, (MultiPolygon, Polygon)):
        st.error("Invalid geometry type")
        return None

    if isinstance(geometry, MultiPolygon):
        geometry = unary_union(geometry)
    
    total_area = geometry.area
    target_area = total_area / n_parts
    
    result = []
    remaining_poly = geometry
    remaining_parts = n_parts

    while remaining_parts > 1:
        try:
            best_angle = analyze_polygon_shape(remaining_poly)
            projection_angle = best_angle + 90
            min_proj, max_proj = get_projection_bounds(remaining_poly, projection_angle)
            offsets = np.linspace(min_proj, max_proj, 40)
            
            best_split = None
            min_area_diff = float('inf')
            best_remaining = None
            
            for offset in offsets:
                split_line = create_split_line(remaining_poly, best_angle, offset)
                try:
                    split_parts = split(remaining_poly, split_line)
                    if not hasattr(split_parts, 'geoms'):
                        continue
                        
                    for part in split_parts.geoms:
                        if not part.is_valid or part.is_empty:
                            continue
                            
                        area_diff = abs(part.area - target_area)
                        if area_diff < min_area_diff:
                            min_area_diff = area_diff
                            best_split = part
                            best_remaining = remaining_poly.difference(part)
                except:
                    continue
            
            if best_split and best_remaining:
                result.append(best_split)
                remaining_poly = best_remaining
                remaining_parts -= 1
            else:
                break
                
        except Exception as e:
            logger.warning(f"Error in splitting: {str(e)}")
            break
    
    if remaining_poly and remaining_poly.is_valid and not remaining_poly.is_empty:
        result.append(remaining_poly)
    
    result = [geom.buffer(0) for geom in result if geom is not None and not geom.is_empty]
    
    max_iterations = 20
    iteration = 0
    while iteration < max_iterations:
        areas = [geom.area for geom in result]
        mean_area = np.mean(areas)
        max_deviation = max(abs(area - mean_area) / mean_area for area in areas)
        
        if max_deviation < 0.001:
            break
            
        adjusted_result = []
        for geom in result:
            area_diff = (mean_area - geom.area) / geom.area
            buffer_amount = np.sign(area_diff) * min(abs(area_diff) * 0.1, 0.001)
            adjusted_geom = geom.buffer(buffer_amount * np.sqrt(geom.area))
            adjusted_geom = adjusted_geom.buffer(0.0001).buffer(-0.0001)
            adjusted_result.append(adjusted_geom)
            
        result = adjusted_result
        iteration += 1
    
    result = [geom.buffer(0.00001).buffer(-0.00001) for geom in result]
    return result


def process_shapefile(input_gdf: gpd.GeoDataFrame, num_subdivisions: int) -> Optional[gpd.GeoDataFrame]:
    """
    Process the input GeoDataFrame to create equal area subdivisions for each polygon.
    """
    try:
        new_geometries = []
        new_attributes = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in input_gdf.iterrows():
            status_text.text(f"Processing polygon {idx+1} of {len(input_gdf)}...")
            geometry = row.geometry
            
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            
            subdivided = split_polygon_equally(geometry, num_subdivisions)
            
            if subdivided:
                total_area = geometry.area
                target_area = total_area / num_subdivisions
                
                for sub_idx, sub_geom in enumerate(subdivided, 1):
                    if not sub_geom.is_empty and sub_geom.is_valid:
                        sub_geom = sub_geom.buffer(0)
                        new_geometries.append(sub_geom)
                        
                        attributes = row.drop("geometry").to_dict()
                        attributes["original_id"] = idx + 1
                        attributes["subdivision_id"] = sub_idx
                        attributes["area"] = sub_geom.area
                        attributes["area_deviation"] = ((sub_geom.area - target_area) / target_area) * 100
                        attributes["target_area"] = target_area
                        attributes["area_difference"] = sub_geom.area - target_area
                        new_attributes.append(attributes)
            
            progress_bar.progress((idx + 1) / len(input_gdf))

        result_gdf = gpd.GeoDataFrame(
            new_attributes, geometry=new_geometries, crs=input_gdf.crs
        )
        
        st.write("\nSubdivision Analysis:")
        summary_df = pd.DataFrame({
            'Original Polygon ID': result_gdf['original_id'],
            'Subdivision ID': result_gdf['subdivision_id'],
            'Area (sq units)': result_gdf['area'].round(4),
            'Target Area (sq units)': result_gdf['target_area'].round(4),
            'Area Difference (sq units)': result_gdf['area_difference'].round(4),
            'Area Deviation (%)': result_gdf['area_deviation'].round(2)
        })
        
        st.write("\nStatistical Summary:")
        stats_df = pd.DataFrame({
            'Metric': ['Mean Area', 'Std Dev Area', 'Max Deviation', 'Min Deviation'],
            'Value': [
                summary_df['Area (sq units)'].mean().round(4),
                summary_df['Area (sq units)'].std().round(4),
                summary_df['Area Deviation (%)'].max().round(2),
                summary_df['Area Deviation (%)'].min().round(2)
            ]
        })
        
        st.write(stats_df)
        st.write("\nDetailed Subdivision Results:")
        st.write(summary_df)
        
        fig = result_gdf.plot(
            column='subdivision_id',
            cmap='tab20',
            figsize=(12, 12),
            edgecolor='black',
            linewidth=0.5
        )
        st.pyplot(fig.figure)
        
        return result_gdf
        
    except Exception as e:
        logger.error(f"Error processing shapefile: {str(e)}")
        st.error(f"Error during processing: {str(e)}")
        raise


def save_gdf_to_shapefile(gdf: gpd.GeoDataFrame, output_dir: str, base_name: str, subdivision_id: int) -> str:
    """
    Save GeoDataFrame to shapefile format with metadata.
    """
    try:
        output_path = os.path.join(output_dir, f"{base_name}_subdivided_{subdivision_id}.shp")
        gdf.to_file(output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error saving shapefile: {str(e)}")
        raise


def main():
    setup_streamlit_page()

    uploaded_file = st.file_uploader(
        "Upload your shapefile",
        type=["zip", "shp"],
        help="Upload a ZIP file containing all shapefile components or a single .shp file",
    )

    if uploaded_file is not None:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                input_dir = os.path.join(temp_dir, "input")
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                if uploaded_file.name.lower().endswith('.zip'):
                    with zipfile.ZipFile(uploaded_file) as zip_ref:
                        is_valid, result = validate_zip_contents(zip_ref)
                        if not is_valid:
                            st.error(result)
                            return
                        base_name = result
                        zip_ref.extractall(input_dir)
                else:
                    base_name = Path(uploaded_file.name).stem
                    with open(os.path.join(input_dir, uploaded_file.name), 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                shp_files = list(Path(input_dir).rglob("*.[sS][hH][pP]"))
                if not shp_files:
                    st.error("No .shp file found!")
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

                st.write("Original Shapefile Preview:")
                st.write(gdf)
                
                fig = gdf.plot(figsize=(10, 10))
                st.pyplot(fig.figure)

                num_subdivisions = st.number_input(
                    "Number of equal-area subdivisions per polygon",
                    min_value=2,
                    max_value=100,
                    value=4,
                    help="Specify how many equal-area parts to create for each polygon",
                )

                if st.button("Process Shapefile"):
                    with st.spinner("Creating equal-area subdivisions..."):
                        result_gdf = process_shapefile(gdf, num_subdivisions)
                        
                        if result_gdf is None:
                            return

                        output_zip_path = os.path.join(temp_dir, f"{base_name}_subdivided.zip")
                        with zipfile.ZipFile(output_zip_path, "w") as zipf:
                            result_gdf.to_file(os.path.join(output_dir, f"{base_name}_subdivided.shp"))
                            for file in Path(output_dir).glob(f"{base_name}_subdivided.*"):
                                zipf.write(file, file.name)

                        with open(output_zip_path, "rb") as f:
                            zip_data = f.read()

                        st.success("Processing complete! Download your results below.")

                        st.download_button(
                            label="Download Subdivided Shapefile",
                            data=zip_data,
                            file_name=f"{base_name}_subdivided.zip",
                            mime="application/zip",
                        )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error("Processing error", exc_info=True)


if __name__ == "__main__":
    main()
