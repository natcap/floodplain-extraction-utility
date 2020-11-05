"""Tracer script to develop floodplain extraction."""
import logging
import os
import shutil

from osgeo import gdal
import pygeoprocessing
import pygeoprocessing.routing
import numpy
import scipy.signal

logging.basicConfig(
    level=logging.INFO,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)

WORKSPACE_DIR = 'workspace'


def scrub_invalid_values(base_array, nodata, new_nodata):
    result = numpy.copy(base_array)
    invalid_mask = (
        ~numpy.isfinite(base_array) |
        numpy.isclose(result, nodata))
    result[invalid_mask] = new_nodata
    return result


def dilate_holes(base_raster_path, target_raster_path):
    """Dialate holes in raster."""
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    base_band = base_raster.GetRasterBand(1)
    nodata = base_band.GetNoDataValue()
    if nodata is None:
        shutil.copyfile(base_raster_path, target_raster_path)
        return

    base_info = pygeoprocessing.get_raster_info(base_raster_path)
    pygeoprocessing.new_raster_from_base(
        base_raster_path, target_raster_path, base_info['datatype'],
        base_info['nodata'])

    target_raster = gdal.OpenEx(
        target_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    target_band = target_raster.GetRasterBand(1)

    n_cols = base_band.XSize
    n_rows = base_band.YSize

    hole_kernel = numpy.array([
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]])
    neighbor_avg_kernel = numpy.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]])

    for offset_dict in pygeoprocessing.iterblocks(
            (base_raster_path, 1), offset_only=True):
        ul_offset_x = 1
        ul_offset_y = 1

        grid_array = numpy.empty(
            (offset_dict['win_ysize']+2, offset_dict['win_xsize']+2))
        grid_array[:] = nodata

        LOGGER.debug(f'**** {offset_dict}')

        if offset_dict['win_xsize']+offset_dict['xoff'] < n_cols:
            offset_dict['win_xsize'] += 1
            LOGGER.debug("if offset_dict['win_xsize']+offset_dict['xoff'] < n_cols:")

        if offset_dict['win_ysize']+offset_dict['yoff'] < n_rows:
            offset_dict['win_ysize'] += 1
            LOGGER.debug("if offset_dict['win_ysize']+offset_dict['yoff'] < n_rows:")

        if offset_dict['xoff'] > 0:
            LOGGER.debug("if offset_dict['xoff'] > 0:")
            offset_dict['xoff'] -= 1
            offset_dict['win_xsize'] += 1
            ul_offset_x -= 1

        if offset_dict['yoff'] > 0:
            LOGGER.debug("if offset_dict['yoff'] > 0:")
            offset_dict['yoff'] -= 1
            offset_dict['win_ysize'] += 1
            ul_offset_y -= 1

        LOGGER.debug(offset_dict)
        LOGGER.debug(ul_offset_x)
        LOGGER.debug(ul_offset_y)
        base_array = base_band.ReadAsArray(**offset_dict)
        LOGGER.debug(base_array.shape)
        LOGGER.debug(grid_array.shape)
        LOGGER.debug(f"{ul_offset_y}:{offset_dict['win_ysize']}")
        LOGGER.debug(f"{ul_offset_x}:{offset_dict['win_xsize']}")
        grid_array[
            ul_offset_y:ul_offset_y+offset_dict['win_ysize'],
            ul_offset_x:ul_offset_x+offset_dict['win_xsize']] = base_array

        nodata_holes = numpy.isclose(grid_array, nodata)
        single_holes = scipy.signal.convolve2d(
            nodata_holes, hole_kernel, mode='same', boundary='fill',
            fillvalue=1) == 9
        neighbor_avg = scipy.signal.convolve2d(
            grid_array, neighbor_avg_kernel, mode='same')
        grid_array[single_holes] = neighbor_avg[single_holes]

        target_band.WriteArray(
            grid_array[
                ul_offset_y:ul_offset_y+offset_dict['win_ysize'],
                ul_offset_x:ul_offset_x+offset_dict['win_xsize']],
            xoff=offset_dict['xoff'],
            yoff=offset_dict['yoff'])
    target_band = None
    target_raster = None



def main():
    """Entry point."""
    try:
        os.makedirs(WORKSPACE_DIR)
    except OSError:
        pass
    #DEM_PATH = 'sample_data/pit_filled_dem.tif'
    DEM_PATH = 'sample_data/Inspring Data/Inputs/DEM/MERIT DEM Pro Agua Purus Acre clip2.tif'

    dem_info = pygeoprocessing.get_raster_info(DEM_PATH)
    dem_type = dem_info['numpy_type']
    scrubbed_dem_path = os.path.join(WORKSPACE_DIR, 'scrubbed_dem.tif')
    nodata = dem_info['nodata'][0]
    new_nodata = float(numpy.finfo(dem_type).min)

    LOGGER.info(f'scrub invalid values to {nodata}')

    # percentile_list = pygeoprocessing.raster_band_percentile(
    #     (DEM_PATH, 1), WORKSPACE_DIR, [1, 99])

    #LOGGER.info(f'percentile_list: {percentile_list}')

    pygeoprocessing.raster_calculator(
        [(DEM_PATH, 1), (nodata, 'raw'), (new_nodata, 'raw')],
        scrub_invalid_values, scrubbed_dem_path,
        dem_info['datatype'], new_nodata)

    LOGGER.info('dialate dem')
    dilated_dem_path = os.path.join(WORKSPACE_DIR, 'dialated_dem.tif')
    dilate_holes(scrubbed_dem_path, dilated_dem_path)

    LOGGER.info('fill pits')
    filled_pits_path = os.path.join(WORKSPACE_DIR, 'filled_pits_dem.tif')
    pygeoprocessing.routing.fill_pits(
        (dilated_dem_path, 1), filled_pits_path)

    flow_dir_d8_path = os.path.join(WORKSPACE_DIR, 'flow_dir_d8.tif')

    LOGGER.info('flow dir d8')
    pygeoprocessing.routing.flow_dir_d8(
        (filled_pits_path, 1), flow_dir_d8_path, working_dir=WORKSPACE_DIR)

    LOGGER.info('flow accum d8')
    flow_accum_d8_path = os.path.join(WORKSPACE_DIR, 'flow_accum_d8.tif')
    pygeoprocessing.routing.flow_accumulation_d8(
        (flow_dir_d8_path, 1), flow_accum_d8_path)




if __name__ == '__main__':
    main()
