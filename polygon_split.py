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


# Function to set up the Streamlit web interface
def setup_streamlit_page():
    """Configure the Streamlit page with title and instructions."""
    # Set the page layout to wide mode
    st.set_page_config(layout="wide")
    # Set the page title
    st.title("Polygon Equal Area Subdivider")
    # Add descriptive text explaining the tool's purpose
    st.write("""
    Upload a shapefile (as a ZIP archive or .shp file) containing polygons to subdivide into equal areas.
    The ZIP file should contain all necessary shapefile components (.shp, .shx, .dbf, .prj).
    Each polygon will be subdivided into equal areas based on the number of subdivisions specified.
    """)


# Function to validate the contents of an uploaded ZIP file
def validate_zip_contents(zip_file) -> Tuple[bool, str]:
    """
    Validate that the uploaded ZIP file contains required shapefile components.
    Returns (is_valid, message_or_basename).
    """
    try:
        # Get list of all files in the ZIP
        file_list = zip_file.namelist()
        # Define required shapefile extensions
        required_extensions = {".shp", ".shx", ".dbf"}
        # Define optional shapefile extensions
        optional_extensions = {".prj", ".cpg", ".qmd"}

        # Get list of actual files (excluding directories)
        found_files = [Path(name) for name in file_list if not name.endswith("/")]
        # Get set of found file extensions
        found_extensions = {f.suffix.lower() for f in found_files}

        # Check if all required extensions are present
        if not required_extensions.issubset(found_extensions):
            missing = required_extensions - found_extensions
            return False, f"Missing required files: {', '.join(missing)}"

        # Check for exactly one .shp file
        shp_files = [f for f in file_list if f.lower().endswith(".shp")]
        if len(shp_files) != 1:
            return False, "ZIP must contain exactly one .shp file"

        # Get the base name and parent directory of the shapefile
        shp_path = Path(shp_files[0])
        parent_dir = str(shp_path.parent) + "/" if shp_path.parent != Path(".") else ""
        base_name = shp_path.stem

        # Verify that all component files have matching names
        for ext in required_extensions | optional_extensions:
            expected_file = f"{parent_dir}{base_name}{ext}"
            if ext in found_extensions and not any(
                f.lower() == expected_file.lower() for f in file_list
            ):
                return False, f"Mismatched filenames: {expected_file} not found"

        return True, base_name
    except Exception as e:
        return False, f"Error validating ZIP contents: {str(e)}"


# Function to analyze the shape of a polygon and determine best split angle
def analyze_polygon_shape(polygon: Union[Polygon, MultiPolygon]) -> float:
    """Analyze polygon shape to determine best split angle"""
    # Get the bounding box coordinates
    bounds = polygon.bounds
    # Calculate width and height
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    
    # Calculate aspect ratio (width/height)
    aspect_ratio = width / height if height != 0 else float('inf')
    
    # Determine best angle based on shape
    if aspect_ratio > 1.5:  # For wide polygons
        return 90  # Use vertical split
    elif aspect_ratio < 0.67:  # For tall polygons
        return 0  # Use horizontal split
    else:
        # For more square-like polygons, analyze different angles
        best_angle = 0
        min_variance = float('inf')
        
        # Try different angles to find the best split
        for angle in [0, 45, 90, 135]:
            split_line = create_split_line(polygon, angle, 0)
            try:
                # Try splitting the polygon at this angle
                parts = split(polygon, split_line)
                if hasattr(parts, 'geoms'):
                    # Calculate area variance for this split
                    areas = [part.area for part in parts.geoms]
                    variance = np.var(areas)
                    # Update best angle if this variance is lower
                    if variance < min_variance:
                        min_variance = variance
                        best_angle = angle
            except:
                continue
                
        return best_angle


# Function to create a line that will split a polygon
def create_split_line(polygon: Union[Polygon, MultiPolygon], angle: float, offset: float) -> LineString:
    """Create a split line at given angle and offset from polygon centroid."""
    # Get polygon bounds
    bounds = polygon.bounds
    # Get polygon center point
    center = polygon.centroid
    
    # Calculate diagonal length to ensure line crosses entire polygon
    diagonal = np.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
    
    # Calculate x and y components based on angle
    dx = np.cos(np.radians(angle)) * diagonal
    dy = np.sin(np.radians(angle)) * diagonal
    
    # Calculate offset perpendicular to split direction
    offset_dx = -dy * offset / diagonal
    offset_dy = dx * offset / diagonal
    
    # Create two points to define the split line
    p1 = Point(center.x + offset_dx - dx, center.y + offset_dy - dy)
    p2 = Point(center.x + offset_dx + dx, center.y + offset_dy + dy)
    
    # Return line connecting the points
    return LineString([p1, p2])


# Function to split a polygon into equal parts
def split_polygon_equally(geometry: Union[Polygon, MultiPolygon], n_parts: int) -> List[Union[Polygon, MultiPolygon]]:
    """
    Split a polygon into n equal parts using an intelligent shape-based approach.
    """
    # If only one part requested, return original geometry
    if n_parts == 1:
        return [geometry]

    # Validate input geometry type
    if not isinstance(geometry, (MultiPolygon, Polygon)):
        st.error("Invalid geometry type")
        return None

    # Convert MultiPolygon to single polygon if possible
    if isinstance(geometry, MultiPolygon):
        geometry = unary_union(geometry)
    
    # Calculate target area for each part
    total_area = geometry.area
    target_area = total_area / n_parts
    
    # Initialize result list and tracking variables
    result = []
    remaining_poly = geometry
    remaining_parts = n_parts

    # Continue splitting until we have desired number of parts
    while remaining_parts > 1:
        try:
            # Determine best split angle based on shape
            best_angle = analyze_polygon_shape(remaining_poly)
            
            # Initialize variables for finding optimal split
            best_split = None
            min_area_diff = float('inf')
            best_remaining = None
            
            # Try different offsets to find optimal split
            for offset_pct in np.linspace(-0.5, 0.5, 40):  # Test 40 different positions
                bounds = remaining_poly.bounds
                offset = offset_pct * (bounds[2] - bounds[0])
                split_line = create_split_line(remaining_poly, best_angle, offset)
                
                try:
                    # Attempt to split polygon
                    split_parts = split(remaining_poly, split_line)
                    if not hasattr(split_parts, 'geoms'):
                        continue
                        
                    # Check each resulting part
                    for part in split_parts.geoms:
                        if not part.is_valid or part.is_empty:
                            continue
                            
                        # Calculate how close this split is to target area
                        area_diff = abs(part.area - target_area)
                        if area_diff < min_area_diff:
                            min_area_diff = area_diff
                            best_split = part
                            best_remaining = remaining_poly.difference(part)
                except:
                    continue
            
            # If we found a valid split, add it to results
            if best_split and best_remaining:
                result.append(best_split)
                remaining_poly = best_remaining
                remaining_parts -= 1
            else:
                break
                
        except Exception as e:
            logger.warning(f"Error in splitting: {str(e)}")
            break
    
    # Add the remaining polygon as final part
    if remaining_poly and remaining_poly.is_valid and not remaining_poly.is_empty:
        result.append(remaining_poly)
    
    # Clean up geometries
    result = [geom.buffer(0) for geom in result if geom is not None and not geom.is_empty]
    
    # Fine-tune areas to make them more equal
    max_iterations = 20
    iteration = 0
    while iteration < max_iterations:
        # Calculate current areas and mean
        areas = [geom.area for geom in result]
        mean_area = np.mean(areas)
        max_deviation = max(abs(area - mean_area) / mean_area for area in areas)
        
        # If deviation is small enough, stop iterating
        if max_deviation < 0.001:
            break
            
        # Adjust each geometry to be closer to mean area
        adjusted_result = []
        for geom in result:
            area_diff = (mean_area - geom.area) / geom.area
            buffer_amount = np.sign(area_diff) * min(abs(area_diff) * 0.1, 0.001)
            adjusted_geom = geom.buffer(buffer_amount * np.sqrt(geom.area))
            # Remove tiny gaps between polygons
            adjusted_geom = adjusted_geom.buffer(0.0001).buffer(-0.0001)
            adjusted_result.append(adjusted_geom)
            
        result = adjusted_result
        iteration += 1
    
    # Final cleanup of geometries
    result = [geom.buffer(0.00001).buffer(-0.00001) for geom in result]
    return result


# Function to process the shapefile and create subdivisions
def process_shapefile(input_gdf: gpd.GeoDataFrame, num_subdivisions: int) -> Optional[gpd.GeoDataFrame]:
    """
    Process the input GeoDataFrame to create equal area subdivisions for each polygon.
    """
    try:
        # Initialize lists for new geometries and attributes
        new_geometries = []
        new_attributes = []
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each polygon in the input
        for idx, row in input_gdf.iterrows():
            status_text.text(f"Processing polygon {idx+1} of {len(input_gdf)}...")
            geometry = row.geometry
            
            # Fix any invalid geometries
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            
            # Split the polygon into equal parts
            subdivided = split_polygon_equally(geometry, num_subdivisions)
            
            if subdivided:
                # Calculate target area
                total_area = geometry.area
                target_area = total_area / num_subdivisions
                
                # Process each subdivided part
                for sub_idx, sub_geom in enumerate(subdivided, 1):
                    if not sub_geom.is_empty and sub_geom.is_valid:
                        # Clean up geometry
                        sub_geom = sub_geom.buffer(0)
                        new_geometries.append(sub_geom)
                        
                        # Create attributes for new polygon
                        attributes = row.drop("geometry").to_dict()
                        attributes["original_id"] = idx + 1
                        attributes["subdivision_id"] = sub_idx
                        attributes["area"] = sub_geom.area
                        attributes["area_deviation"] = ((sub_geom.area - target_area) / target_area) * 100
                        attributes["target_area"] = target_area
                        attributes["area_difference"] = sub_geom.area - target_area
                        new_attributes.append(attributes)
            
            # Update progress bar
            progress_bar.progress((idx + 1) / len(input_gdf))

        # Create new GeoDataFrame with results
        result_gdf = gpd.GeoDataFrame(
            new_attributes, geometry=new_geometries, crs=input_gdf.crs
        )
        
        # Display subdivision analysis
        st.write("\nSubdivision Analysis:")
        summary_df = pd.DataFrame({
            'Original Polygon ID': result_gdf['original_id'],
            'Subdivision ID': result_gdf['subdivision_id'],
            'Area (sq units)': result_gdf['area'].round(4),
            'Target Area (sq units)': result_gdf['target_area'].round(4),
            'Area Difference (sq units)': result_gdf['area_difference'].round(4),
            'Area Deviation (%)': result_gdf['area_deviation'].round(2)
        })
        
        # Calculate and display statistical summary
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
        
        # Display results
        st.write(stats_df)
        st.write("\nDetailed Subdivision Results:")
        st.write(summary_df)
        
        # Create and display map visualization
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
        # Log and display any errors
        logger.error(f"Error processing shapefile: {str(e)}")
        st.error(f"Error during processing: {str(e)}")
        raise


# Function to save processed data to a shapefile
def save_gdf_to_shapefile(gdf: gpd.GeoDataFrame, output_dir: str, base_name: str, subdivision_id: int) -> str:
    """
    Save GeoDataFrame to shapefile format with metadata.
    """
    try:
        # Create output path with unique name
        output_path = os.path.join(output_dir, f"{base_name}_subdivided_{subdivision_id}.shp")
        # Save GeoDataFrame to shapefile
        gdf.to_file(output_path)
        return output_path
    except Exception as e:
        # Log any errors during save
        logger.error(f"Error saving shapefile: {str(e)}")
        raise


# Main function that runs the application
def main():
    # Set up the Streamlit interface
    setup_streamlit_page()

    # Create file uploader widget
    uploaded_file = st.file_uploader(
        "Upload your shapefile",
        type=["zip", "shp"],
        help="Upload a ZIP file containing all shapefile components or a single .shp file",
    )

    # Process uploaded file if present
    if uploaded_file is not None:
        try:
            # Create temporary directories for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                input_dir = os.path.join(temp_dir, "input")
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(input_dir, exist_ok=True)
                os.makedirs(output_dir, exist_ok=True)

                # Handle ZIP file upload
                if uploaded_file.name.lower().endswith('.zip'):
                    with zipfile.ZipFile(uploaded_file) as zip_ref:
                        # Validate ZIP contents
                        is_valid, result = validate_zip_contents(zip_ref)
                        if not is_valid:
                            st.error(result)
                            return
                        base_name = result
                        zip_ref.extractall(input_dir)
                # Handle direct shapefile upload
                else:
                    base_name = Path(uploaded_file.name).stem
                    with open(os.path.join(input_dir, uploaded_file.name), 'wb') as f:
                        f.write(uploaded_file.getbuffer())

                # Find shapefile in input directory
                shp_files = list(Path(input_dir).rglob("*.[sS][hH][pP]"))
                if not shp_files:
                    st.error("No .shp file found!")
                    return

                shp_file = shp_files[0]

                # Try reading shapefile with different encodings
                try:
                    gdf = gpd.read_file(shp_file, encoding="utf-8")
                except Exception:
                    try:
                        gdf = gpd.read_file(shp_file, encoding="latin1")
                    except Exception as e:
                        st.error(f"Failed to read shapefile: {str(e)}")
                        raise

                # Display original data preview
                st.write("Original Shapefile Preview:")
                st.write(gdf)
                
                # Display original map
                fig = gdf.plot(figsize=(10, 10))
                st.pyplot(fig.figure)

                # Create input for number of subdivisions
                num_subdivisions = st.number_input(
                    "Number of equal-area subdivisions per polygon",
                    min_value=2,
                    max_value=100,
                    value=4,
                    help="Specify how many equal-area parts to create for each polygon",
                )

                # Process button and handling
                if st.button("Process Shapefile"):
                    with st.spinner("Creating equal-area subdivisions..."):
                        # Process the shapefile
                        result_gdf = process_shapefile(gdf, num_subdivisions)
                        
                        if result_gdf is None:
                            return

                        # Save results to ZIP file
                        output_zip_path = os.path.join(temp_dir, f"{base_name}_subdivided.zip")
                        with zipfile.ZipFile(output_zip_path, "w") as zipf:
                            result_gdf.to_file(os.path.join(output_dir, f"{base_name}_subdivided.shp"))
                            for file in Path(output_dir).glob(f"{base_name}_subdivided.*"):
                                zipf.write(file, file.name)

                        # Read ZIP file for download
                        with open(output_zip_path, "rb") as f:
                            zip_data = f.read()

                        # Show success message
                        st.success("Processing complete! Download your results below.")

                        # Create download button
                        st.download_button(
                            label="Download Subdivided Shapefile",
                            data=zip_data,
                            file_name=f"{base_name}_subdivided.zip",
                            mime="application/zip",
                        )

        except Exception as e:
            # Handle and display any errors
            st.error(f"An error occurred: {str(e)}")
            logger.error("Processing error", exc_info=True)


# Run main function if script is run directly
if __name__ == "__main__":
    main()
