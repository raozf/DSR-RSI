"""Robust table-based affine depth fit with disjoint spatial validation.

The residual describes visible table consistency, never object-depth accuracy.
Small random minimal sets let RANSAC reject object/robot pixels. Taking 25% of
all pixels per hypothesis contaminates almost every hypothesis in clutter.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor


class DepthCalibrationError(ValueError):
    pass


def visible_table_support(masks, shape):
    """Select table interiors from independent RGB segmentation, never depth residuals."""
    import cv2
    masks=np.asarray(masks)
    if masks.dtype!=np.bool_ or masks.ndim!=3 or masks.shape[1:]!=tuple(shape) or not len(masks):
        raise DepthCalibrationError('table_rgb_support_unresolved')
    # Remove mixed foreground/background depth at object and table boundaries.
    return cv2.erode(np.any(masks,axis=0).astype(np.uint8),np.ones((7,7),np.uint8))>0


def calibrate(depth, expected_forward, eligible):
    depth=np.asarray(depth,float);expected=np.asarray(expected_forward,float);mask=np.asarray(eligible,bool)
    if depth.shape!=expected.shape or mask.shape!=depth.shape or depth.ndim!=2:
        raise DepthCalibrationError('depth_calibration_shape')
    mask=mask & np.isfinite(depth) & np.isfinite(expected) & (depth>0) & (expected>0)
    yy,xx=np.nonzero(mask)
    if len(xx)<400:raise DepthCalibrationError('depth_calibration_support')
    x,y=depth[mask],expected[mask]
    # A deterministic spatial split, unrelated to outcomes, seeds or layouts.
    validation=((xx//4+yy//4)%3)==0
    if min(validation.sum(),(~validation).sum())<100 or np.ptp(y)<.12 or np.ptp(x)<1e-5:
        raise DepthCalibrationError('depth_calibration_unobservable_scale')
    model=RANSACRegressor(estimator=LinearRegression(),min_samples=8,
        residual_threshold=.015,max_trials=200,random_state=0)
    try:model.fit(x[~validation,None],y[~validation])
    except ValueError as error:raise DepthCalibrationError('depth_calibration_no_consensus') from error
    slope=float(model.estimator_.coef_[0]);offset=float(model.estimator_.intercept_)
    if slope<=0:raise DepthCalibrationError('depth_calibration_reversed_metric_depth')
    residual=abs(model.predict(x[:,None])-y)
    good=residual<=.015
    heldout_fraction=float(np.mean(good[validation]));heldout_median=float(np.median(residual[validation]))
    quadrants=[]
    for left in (True,False):
        for upper in (True,False):
            region=validation & ((xx<depth.shape[1]/2)==left) & ((yy<depth.shape[0]/2)==upper)
            fraction=float(np.mean(good[region])) if region.sum()>=20 else 0.
            quadrants.append(fraction)
    if heldout_fraction<.60 or heldout_median>.015 or min(quadrants)<.35:
        raise DepthCalibrationError('depth_calibration_residual')
    distance=model.predict(depth.reshape(-1,1)).reshape(depth.shape)
    if not np.isfinite(distance).all():raise DepthCalibrationError('depth_calibration_nonfinite_image')
    receipt=dict(training_points=int((~validation).sum()),validation_points=int(validation.sum()),
        slope=slope,offset=offset,heldout_inlier_fraction=heldout_fraction,
        heldout_median_error_m=heldout_median,heldout_quadrant_inlier_fractions=quadrants,
        residual_limit_m=.015,object_depth_accuracy='unqualified')
    return distance,receipt
