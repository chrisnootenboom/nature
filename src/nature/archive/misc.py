import os
import shutil
import pandas as pd
import geopandas as gpd
from pathlib import Path
from osgeo import ogr, gdal
import census
import us
import pygeoprocessing
import logging
import warnings
import requests

import rasterio as rio


def create_shapefile(layer, feature_name, folder):
    """Export selected feature to shapefiles.

    Parameters:
        layer (layer): GDAL layer (typically a subset of a larger layer) that will be exported to shapefile.
        feature_name (string): the name of the eventual shapefile.
        folder (string): path to output folder.

    Returns:
        File path to the exported shapefile.

    Raises:
        None
    """

    # Create Shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = Path(folder) / f"{feature_name}.shp"
    if shapefile.exists():
        driver.DeleteDataSource(str(shapefile))
    shapefile_datasource = driver.CreateDataSource(str(shapefile))

    # Extract spatial reference and layer definition
    spatial_reference = layer.GetSpatialRef()
    inLayerDefn = layer.GetLayerDefn()

    shapefile_layer = shapefile_datasource.CreateLayer(
        os.path.splitext(os.path.split(shapefile)[1])[0],  # layer name
        spatial_reference,
        inLayerDefn.GetGeomType(),
    )

    # Add input Layer Fields to the output Layer if it is the one we want
    for i in range(0, inLayerDefn.GetFieldCount()):
        shapefile_layer.CreateField(inLayerDefn.GetFieldDefn(i))

    # Add features to the ouput Layer
    outLayerDefn = shapefile_layer.GetLayerDefn()
    for inFeature in layer:
        # Create output Feature
        shapefile_feature = ogr.Feature(outLayerDefn)

        # Add field values from input Layer
        for i in range(0, outLayerDefn.GetFieldCount()):
            shapefile_feature.SetField(
                outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i)
            )

        # Add new feature to output Layer
        geom = inFeature.GetGeometryRef()
        shapefile_feature.SetGeometry(geom.Clone())
        shapefile_layer.CreateFeature(shapefile_feature)
        shapefile_feature = None

    # Save and close DataSources
    shapefile_datasource = None

    return shapefile


def create_shapefile_and_buffer(layer, feature_name, folder, distance):
    """Export selected feature (and associated buffer feature) to shapefiles.

    Parameters:
        layer (layer): GDAL layer (typically a subset of a larger layer) that will be exported to shapefile.
        feature_name (string): the name of the eventual shapefile.
        folder (string): path to output folder.
        distance (integer): the distance (in meters) to buffer the shapefile.

    Returns:
        File paths to the exported shapefile and buffered shapefile as a tuple.

    Raises:
        None
    """

    # Create Shapefile
    driver = ogr.GetDriverByName("ESRI Shapefile")
    shapefile = Path(folder) / f"{feature_name}.shp"
    if shapefile.exists():
        driver.DeleteDataSource(str(shapefile))
    shapefile_datasource = driver.CreateDataSource(str(shapefile))

    # Extract spatial reference and layer definition
    spatial_reference = layer.GetSpatialRef()
    inLayerDefn = layer.GetLayerDefn()

    shapefile_layer = shapefile_datasource.CreateLayer(
        os.path.splitext(os.path.split(shapefile)[1])[0],  # layer name
        spatial_reference,
        inLayerDefn.GetGeomType(),
    )
    # Buffer
    buffer = Path(folder) / f"{feature_name}_buffer.shp"
    if buffer.exists():
        driver.DeleteDataSource(str(buffer))
    buffer_datasource = driver.CreateDataSource(str(buffer))
    buffer_layer = buffer_datasource.CreateLayer(
        os.path.splitext(os.path.split(shapefile)[1])[0],  # layer name
        spatial_reference,
        inLayerDefn.GetGeomType(),
    )

    # Add input Layer Fields to the output Layer if it is the one we want
    for i in range(0, inLayerDefn.GetFieldCount()):
        shapefile_layer.CreateField(inLayerDefn.GetFieldDefn(i))
        buffer_layer.CreateField(inLayerDefn.GetFieldDefn(i))

    # Add features to the ouput Layer
    outLayerDefn = shapefile_layer.GetLayerDefn()
    for inFeature in layer:
        # Create output Feature
        shapefile_feature = ogr.Feature(outLayerDefn)
        buffer_feature = ogr.Feature(outLayerDefn)

        # Add field values from input Layer
        for i in range(0, outLayerDefn.GetFieldCount()):
            shapefile_feature.SetField(
                outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i)
            )
            buffer_feature.SetField(
                outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i)
            )

        # Add new feature to output Layer
        geom = inFeature.GetGeometryRef()
        shapefile_feature.SetGeometry(geom.Clone())
        shapefile_layer.CreateFeature(shapefile_feature)
        shapefile_feature = None

        buffer_feature.SetGeometry(geom.Clone().Buffer(float(distance)))
        buffer_layer.CreateFeature(buffer_feature)
        buffer_feature = None

    # Save and close DataSources
    shapefile_datasource = None
    buffer_datasource = None

    return shapefile, buffer
