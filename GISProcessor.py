import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, MultiLineString
from shapely.vectorized import contains
import os
import matplotlib.pyplot as plt

class GISProcessor:
    def __init__(self, dem_path):
        self.dem = rasterio.open(dem_path)
        self.dem_array = self.dem.read(1)
        self.dem_transform = self.dem.transform
        print(f"DEM CRS: {self.dem.crs}")
        print(f"DEM bounds: {self.dem.bounds}")
        print(f"DEM shape: {self.dem_array.shape}")

    def process_shapefile(self, shapefile):
        result = np.zeros(self.dem_array.shape, dtype=np.uint8)
        
        print(f"Processing shapefile with CRS: {shapefile.crs}")
        print(f"Shapefile bounds: {shapefile.total_bounds}")
        
        if shapefile.crs != self.dem.crs:
            shapefile = shapefile.to_crs(self.dem.crs)
        
        rows, cols = self.dem_array.shape
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))
        x_coords, y_coords = self.dem_transform * (x_coords, y_coords)
        
        for geometry in shapefile.geometry:
            if geometry.is_empty:
                continue
            geom_type = geometry.geom_type
            print(f"Processing geometry of type: {geom_type}")
            if geom_type in ["LineString", "MultiLineString"]:
                buffered_geom = geometry.buffer(1)
            else:
                buffered_geom = geometry
            
            mask = contains(buffered_geom, x_coords, y_coords)
            result[mask] = 1

            # 추가: 불연속적인 부분을 연결하기 위해 작은 간격을 채우기
            if geom_type in ["LineString", "MultiLineString"]:
                for i in range(rows):
                    for j in range(cols):
                        if result[i, j] == 1:
                            # 주변 8방향을 확인하여 1로 설정
                            for di in [-1, 0, 1]:
                                for dj in [-1, 0, 1]:
                                    ni, nj = i + di, j + dj
                                    if 0 <= ni < rows and 0 <= nj < cols:
                                        result[ni, nj] = 1
        
        print(f"Processed shapefile. Non-zero elements: {np.count_nonzero(result)}")
        return result

    def load_or_process_array(self, filename, shapefile):
        if os.path.exists(filename):
            print(f"Loading existing array from {filename}")
            array = np.load(filename)
        else:
            print(f"Processing and saving array to {filename}")
            array = self.process_shapefile(shapefile)
            np.save(filename, array)
        self.check_non_zero_values(array, filename)
        return array

    def check_non_zero_values(self, array, filename):
        non_zero_count = np.count_nonzero(array)
        print(f"Number of non-zero elements in {filename}: {non_zero_count}")
        if non_zero_count > 0:
            print(f"Non-zero elements found in {filename}.")
        else:
            print(f"No non-zero elements found in {filename}.")

    def create_featured_dem(self, rirsv, wkmstrm, climbpath, road, watershed, channels):
        dem_slice = self.dem_array
        
        rirsv_array = self.load_or_process_array('rirsv_array.npy', rirsv)
        wkmstrm_array = self.load_or_process_array('wkmstrm_array.npy', wkmstrm)
        climbpath_array = self.load_or_process_array('climbpath_array.npy', climbpath)
        road_array = self.load_or_process_array('road_array.npy', road)
        watershed_array = self.load_or_process_array('watershed_array.npy', watershed)
        channels_array = self.load_or_process_array('channels_array.npy', channels)

        combined_array = np.stack((dem_slice, rirsv_array, climbpath_array, wkmstrm_array, road_array, watershed_array, channels_array), axis=-1)
        print("Combined array shape:", combined_array.shape)

        return combined_array

def load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, climbpath_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file, area_difference_file):
    rirsv = gpd.read_file(rirsv_shp_file)
    wkmstrm = gpd.read_file(wkmstrm_shp_file)
    climbpath = gpd.read_file(climbpath_shp_file)
    road = gpd.read_file(road_shp_file)
    watershed_basins = gpd.read_file(watershed_basins_shp_file)
    channels = gpd.read_file(channels_shp_file)
    test_area = gpd.read_file(area_difference_file)
    
    for name, shapefile in zip(["RIRSV", "WKMSTRM", "Climbpath", "Road", "Watershed", "Channels", "Test Area"],
                               [rirsv, wkmstrm, climbpath, road, watershed_basins, channels, test_area]):
        print(f"{name} Shapefile CRS: {shapefile.crs}")
        print(f"{name} Shapefile bounds: {shapefile.total_bounds}")
    
    return rirsv, wkmstrm, climbpath, road, watershed_basins, channels, test_area

def visualize_array(array, title, shapefile=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(array, cmap='gray')
    if shapefile is not None:
        shapefile.plot(ax=plt.gca(), color='red')
    plt.title(title)
    plt.colorbar()
    plt.show()

def visualize_watershed_array(npy_file):
    """
    Load and visualize a single watershed array from an npy file.
    
    Parameters:
    npy_file (str): Path to the npy file containing the watershed array.
    """
    # Load the array from the npy file
    watershed_array = np.load(npy_file)
    
    # Visualize the array
    plt.figure(figsize=(10, 10))
    plt.imshow(watershed_array, cmap='gray')
    plt.title('Watershed Array')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    from config import *
    
    processor = GISProcessor(dem_file)
    rirsv, wkmstrm, climbpath, road, watershed_basins, channels, test_area = load_shapefiles(rirsv_shp_file, wkmstrm_shp_file, climbpath_shp_file, road_shp_file, watershed_basins_shp_file, channels_shp_file, area_difference_file)

    # Create featured DEM
    print("Creating featured DEM")
    featured_dem = processor.create_featured_dem(rirsv, wkmstrm, climbpath, road, watershed_basins, channels)
    print("Featured DEM shape:", featured_dem.shape)

    # Save the array
    save_path = 'featured_dem.npy'
    np.save(save_path, featured_dem)
    print(f"Featured DEM saved to {save_path}")

    # Process test area
    test_area_result = processor.process_shapefile(test_area)
    np.save('test_area_result.npy', test_area_result)
    print("Test area result shape:", test_area_result.shape)

    # Visualize results
    # visualize_array(featured_dem[:,:,0], 'DEM')
    # visualize_array(featured_dem[:,:,1], 'RIRSV', rirsv)
    # visualize_array(featured_dem[:,:,2], 'Climbpath', climbpath)
    # visualize_array(featured_dem[:,:,3], 'WKMSTRM', wkmstrm)
    # visualize_array(featured_dem[:,:,4], 'Road', road)
    # visualize_array(featured_dem[:,:,5], 'Watershed', watershed_basins)
    # visualize_array(featured_dem[:,:,6], 'Channels', channels)
    # visualize_array(test_area_result, 'Test Area', test_area)
    visualize_watershed_array('/Users/heekim/Desktop/heekimjun/SAR_Project_Agent/watershed_array.npy')