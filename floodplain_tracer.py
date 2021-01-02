"""Tracer script to develop floodplain extraction."""
import collections
import logging
import os
import shutil
import sys
import multiprocessing
import threading

from osgeo import gdal
import pygeoprocessing
import pygeoprocessing.routing
import numpy
import matplotlib.pyplot
import scipy.signal
import taskgraph

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
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
    # try:
    #     os.makedirs(WORKSPACE_DIR)
    # except OSError:
    #     pass
    #DEM_PATH = 'sample_data/pit_filled_dem.tif'
    DEM_PATH = 'sample_data/Inspring Data/Inputs/DEM/MERIT DEM Pro Agua Purus Acre clip2.tif'

    dem_info = pygeoprocessing.get_raster_info(DEM_PATH)
    dem_type = dem_info['numpy_type']
    scrubbed_dem_path = os.path.join(WORKSPACE_DIR, 'scrubbed_dem.tif')
    nodata = dem_info['nodata'][0]
    new_nodata = float(numpy.finfo(dem_type).min)

    LOGGER.info(f'scrub invalid values to {nodata}')

    # percentile_list = pygeoprocessing.raster_band_percentile(
    # #     (DEM_PATH, 1), WORKSPACE_DIR, [1, 99])

    # #LOGGER.info(f'percentile_list: {percentile_list}')
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)
    scrub_dem_task = task_graph.add_task(
        func=pygeoprocessing.raster_calculator,
        args=(
            [(DEM_PATH, 1), (nodata, 'raw'), (new_nodata, 'raw')],
            scrub_invalid_values, scrubbed_dem_path,
            dem_info['datatype'], new_nodata),
        target_path_list=[scrubbed_dem_path],
        task_name='scrub dem')

    # LOGGER.info('dialate dem')
    # dilated_dem_path = os.path.join(WORKSPACE_DIR, 'dialated_dem.tif')
    # dilate_holes(scrubbed_dem_path, dilated_dem_path)

    LOGGER.info('fill pits')
    filled_pits_path = os.path.join(WORKSPACE_DIR, 'filled_pits_dem.tif')
    fill_pits_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=((scrubbed_dem_path, 1), filled_pits_path),
        target_path_list=[filled_pits_path],
        dependent_task_list=[scrub_dem_task],
        task_name='fill pits')

    # slope_path = os.path.join(WORKSPACE_DIR, 'slope.tif')
    # pygeoprocessing.calculate_slope((DEM_PATH, 1), slope_path)

    LOGGER.info('flow dir d8')
    flow_dir_d8_path = os.path.join(WORKSPACE_DIR, 'flow_dir_d8.tif')
    flow_dir_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_d8,
        args=((filled_pits_path, 1), flow_dir_d8_path),
        kwargs={'working_dir': WORKSPACE_DIR},
        target_path_list=[flow_dir_d8_path],
        dependent_task_list=[fill_pits_task],
        task_name='flow dir d8')

    LOGGER.info('flow accum d8')
    flow_accum_d8_path = os.path.join(WORKSPACE_DIR, 'flow_accum_d8.tif')
    flow_accum_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_accumulation_d8,
        args=((flow_dir_d8_path, 1), flow_accum_d8_path),
        target_path_list=[flow_accum_d8_path],
        dependent_task_list=[flow_dir_task],
        task_name='flow accum d8')

    flow_threshold = 100
    stream_vector_path = os.path.join(
        WORKSPACE_DIR, f'stream_segments_{flow_threshold}.gpkg')
    extract_stream_task = task_graph.add_task(
        func=pygeoprocessing.routing.extract_strahler_streams_d8,
        args=(
            (flow_dir_d8_path, 1), (flow_accum_d8_path, 1),
            (filled_pits_path, 1), stream_vector_path),
        kwargs={'min_flow_accum_threshold': flow_threshold, 'river_order': 7},
        target_path_list=[stream_vector_path],
        hash_target_files=False,
        dependent_task_list=[flow_accum_task],
        task_name='stream extraction')

    target_watershed_boundary_vector_path = os.path.join(
        WORKSPACE_DIR, 'watershed_boundary.gpkg')
    calculate_watershed_boundary_task = task_graph.add_task(
        func=pygeoprocessing.routing.calculate_watershed_boundary,
        args=(
            (flow_dir_d8_path, 1), stream_vector_path,
            target_watershed_boundary_vector_path, -100),
        target_path_list=[target_watershed_boundary_vector_path],
        transient_run=True,
        dependent_task_list=[extract_stream_task],
        task_name='watershed boundary')

    # river_id = 338
    # stream_vector = gdal.OpenEx(stream_vector_path, gdal.OF_VECTOR)
    # stream_layer = stream_vector.GetLayer()
    # drop_distance_collection = collections.defaultdict(list)
    # stream_layer.SetAttributeFilter(f'"river_id"={river_id}')
    # for stream_feature in stream_layer:
    #     drop_distance_collection[stream_feature.GetField('order')].append(
    #         stream_feature.GetField('drop_distance'))

    #     fig, ax = matplotlib.pyplot.subplots()

    #     t_stat, p_val = scipy.stats.ttest_ind(
    #         drop_distance_collection[1],
    #         drop_distance_collection[2], equal_var=True)

    #     ax.set_title(
    #         f'Drop Distance vs Order for flow thresh {flow_threshold}\n'
    #         f't={t_stat:.3f} p={p_val:.3f}')
    #     ax.set_ylim([0, 150])
    #     ax.boxplot([
    #         drop_distance_collection[order]
    #         for order in sorted(drop_distance_collection)])

    # matplotlib.pyplot.show()


if __name__ == '__main__':
    main()
