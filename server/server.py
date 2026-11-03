"""
Spatter MCP Server - Using trained polynomial models for predictions

Model Input/Output Scales:
- Depth, Width, Length, Area, Volume models: Input (P in W, V in mm/s), Output in µm
- Spatter model: Input (P, V) with normalized features [V/P, V³, V⁴, P³], Output in µm³

New in v3.2:
- Fixed mcp_find_best and mcp_optimize by using internal implementation functions
- Added _filtered_grid_scan_impl() and _find_best_impl() to avoid @mcp.tool() decorator issues
- MCP tools now delegate to internal functions for proper function calls

Version: 3.2 - Fixed internal tool calling issues
"""

import json
import ast
import os
import sys
import uuid
import numpy as np
import warnings
from typing import Optional, List, Any, Union, Dict
from fastmcp import FastMCP

# Suppress sklearn version warnings
warnings.filterwarnings('ignore')

# Initialize the MCP server
mcp = FastMCP("spatter_mcp")


# =============================================================================
# MATERIAL CONFIGURATION - UPDATE THESE TO MATCH YOUR ACTUAL DATASET
# =============================================================================

MATERIAL_BOUNDS = {
    "SS316L": {
        "P_range": [160.0, 430.0],
        "V_range": [100.0, 1500.0],
        "J_range": [0.11, 4.30],
        "P_typical": [180.0, 380.0],
        "V_typical": [200.0, 1200.0],
        "data_points": 279,
        "source": "279 FLOW-3D single-track simulations (SS316L)",
        "units": {
            "P": "Watts",
            "V": "mm/s",
            "J": "J/mm",
            "depth": "µm",
            "width": "µm",
            "length": "µm",
            "spatter_volume": "µm³"
        }
    },
}

DEFAULT_MATERIAL = "SS316L"

# Defect classification thresholds
KEYHOLE_DW_THRESHOLD = 1.25
LOF_DEPTH_THRESHOLD = 137
BALLING_LW_MIN = 2.35
BALLING_LW_MAX = 3.7


# =============================================================================
# MODEL LOADING
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", SCRIPT_DIR)

_models = {}


class SurrogateUnavailable(RuntimeError):
    """Raised when a surrogate model is missing or fails; the server then returns an error instead of a value."""


def load_models():
    """Load all polynomial models from .pkl files."""
    global _models
    
    if _models:
        return _models
    
    try:
        import joblib
    except ImportError:
        print("Error: joblib not available; predictions will fail with SurrogateUnavailable", file=sys.stderr)
        return {}
    
    model_names = ['Depth', 'Width', 'Length', 'Area', 'Volume', 'Spatter']

    for name in model_names:
        model_path = os.path.join(MODEL_DIR, f"{name}_poly_model.pkl")
        try:
            _models[name] = joblib.load(model_path)
            print(f"Loaded {name} model from {model_path}", file=sys.stderr)
        except FileNotFoundError:
            print(f"Warning: {model_path} not found", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load {name} model: {e}", file=sys.stderr)
    
    return _models


def get_models():
    """Get loaded models, loading them if necessary."""
    global _models
    if not _models:
        load_models()
    return _models


# =============================================================================
# CUSTOM TYPE FOR ARRAY PARAMETERS
# =============================================================================

ArrayParam = Union[List[float], str, None]


# =============================================================================
# PARAMETER PARSING HELPERS
# =============================================================================

def parse_array_param(value: Any) -> Optional[List[float]]:
    """Parse array parameter from various formats."""
    if value is None:
        return None
    
    if isinstance(value, (list, tuple)):
        return [float(x) if x is not None else None for x in value]
    
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [float(x) if x is not None else None for x in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [float(x) if x is not None else None for x in parsed]
        except (ValueError, SyntaxError):
            pass
        
        try:
            parts = [x.strip() for x in value.split(',')]
            return [float(x) if x.lower() not in ('none', 'null', '') else None for x in parts]
        except ValueError:
            pass
    
    return None


def calculate_bounds_from_j_limits(
    j_limits: List[float],
    keep_P: Optional[float] = None,
    keep_V: Optional[float] = None,
    current_P_bounds: Optional[List[float]] = None,
    current_V_bounds: Optional[List[float]] = None
) -> tuple:
    """Auto-calculate P or V bounds from j_limits."""
    J_min, J_max = j_limits
    P_bounds = current_P_bounds
    V_bounds = current_V_bounds
    
    if keep_P is not None and current_V_bounds is None:
        V_from_J_max = keep_P / J_max
        V_from_J_min = keep_P / J_min
        V_bounds = [V_from_J_max, V_from_J_min]
        
    elif keep_V is not None and current_P_bounds is None:
        P_from_J_min = J_min * keep_V
        P_from_J_max = J_max * keep_V
        P_bounds = [P_from_J_min, P_from_J_max]
    
    elif current_P_bounds is not None and current_V_bounds is None:
        P_min, P_max = current_P_bounds
        V_min_from_J = P_min / J_max
        V_max_from_J = P_max / J_min
        V_bounds = [V_min_from_J, V_max_from_J]
        
    elif current_V_bounds is not None and current_P_bounds is None:
        V_min, V_max = current_V_bounds
        P_min_from_J = J_min * V_min
        P_max_from_J = J_max * V_max
        P_bounds = [P_min_from_J, P_max_from_J]
    
    return P_bounds, V_bounds


# =============================================================================
# MODEL FUNCTIONS
# =============================================================================

def get_material_ranges(material: str) -> dict:
    """Get P,V ranges for a material. Only materials with trained models are accepted;
    any other name raises instead of silently returning SS316L predictions."""
    key = next((m for m in MATERIAL_BOUNDS if m.lower() == str(material or DEFAULT_MATERIAL).strip().lower()), None)
    if key is None:
        raise ValueError(f"Material '{material}' is not supported: the surrogate models are trained only for "
                         f"{', '.join(MATERIAL_BOUNDS)}. No prediction is available for this material.")
    bounds = MATERIAL_BOUNDS[key]
    return {
        "P": bounds["P_range"],
        "V": bounds["V_range"]
    }


def predict_geometry(P: float, V: float) -> dict:
    """
    Predict melt pool geometry using polynomial models.
    
    Models expect: P (W), V (mm/s)
    Models output: dimensions in µm
    """
    models = get_models()
    X = np.array([[P, V]])
    
    results = {}
    
    for name in ['Depth', 'Width', 'Length', 'Area', 'Volume']:
        if name in models:
            try:
                results[name.lower()] = float(models[name].predict(X)[0])
            except Exception as e:
                print(f"Prediction error for {name}: {e}", file=sys.stderr)
                results[name.lower()] = None
        else:
            results[name.lower()] = None
    
    # No fallback: a missing or failing surrogate must never be replaced by invented values.
    missing = [k for k, v in results.items() if v is None]
    if missing:
        raise SurrogateUnavailable(
            f"Surrogate model(s) for {', '.join(missing)} could not be loaded or evaluated; "
            "no prediction is returned. Check the *_poly_model.pkl files and the scikit-learn version.")
    
    return results


def predict_spatter_volume_from_PV(P: float, V: float) -> float:
    """
    Predict spatter volume using polynomial model.
    
    Model is a bundle containing:
    - 'model': sklearn Pipeline (PolynomialFeatures + LinearRegression)
    - 'P90': normalization constant for Power
    - 'V90': normalization constant for Velocity
    
    Features (normalized): ['V/P', 'V^3', 'V^4', 'P^3']
    Output: spatter volume in µm³
    """
    models = get_models()
    
    if 'Spatter' in models:
        try:
            bundle = models['Spatter']
            
            if isinstance(bundle, dict):
                sklearn_model = bundle['model']
                P90 = bundle['P90']
                V90 = bundle['V90']
                
                eps = 1e-12
                P_safe = max(P, eps)
                V_safe = max(V, eps)
                
                theta_P = P_safe / P90
                theta_V = V_safe / V90
                
                V_div_P = theta_V / theta_P
                V_cu = theta_V ** 3
                V_qu = theta_V ** 4
                P_cu = theta_P ** 3
                
                X = np.array([[V_div_P, V_cu, V_qu, P_cu]])
                
                spatter = sklearn_model.predict(X)[0]
                return float(max(0, spatter))
            else:
                X = np.array([[P, V]])
                return float(max(0, bundle.predict(X)[0]))
                
        except Exception as e:
            print(f"Spatter prediction error: {e}", file=sys.stderr)
    
    # No fallback: never return an invented spatter value.
    raise SurrogateUnavailable(
        "Spatter surrogate (Spatter_poly_model.pkl) could not be loaded or evaluated; no prediction is returned.")


def predict_spatter(material: str, P: float, V: float, extra: dict = None) -> dict:
    """Predict spatter volume for given parameters (inputs clamped to the trained range)."""
    ranges = get_material_ranges(material)
    P_in, V_in = P, V
    P = max(ranges["P"][0], min(ranges["P"][1], P))
    V = max(ranges["V"][0], min(ranges["V"][1], V))
    J = P / V
    spatter = predict_spatter_volume_from_PV(P, V)
    
    return {
        "material": material,
        "P": P,
        "V": V,
        "J": J,
        "spatter_volume": spatter,
        "meta": {
            "input": {"P": P_in, "V": V_in, "J": P_in / V_in if V_in else None},
            "used": {"P": P, "V": V},
            "ranges": {"P": ranges["P"], "V": ranges["V"]},
            "clamped": {"P": P_in != P, "V": V_in != V},
        },
    }


def predict_all_metrics(material: str, P: float, V: float, extra: dict = None) -> dict:
    """Predict all metrics including melt pool geometry and defects."""
    ranges = get_material_ranges(material)
    
    P_clamped = max(ranges["P"][0], min(ranges["P"][1], P))
    V_clamped = max(ranges["V"][0], min(ranges["V"][1], V))
    # Evaluate the surrogates only at the clamped point, so the polynomials are never extrapolated.
    P_in, V_in, J_in = P, V, (P / V if V else None)
    P, V = P_clamped, V_clamped
    
    J = P / V
    
    geom = predict_geometry(P, V)
    depth = geom['depth']
    width = geom['width']
    length = geom['length']
    area = geom['area']
    volume = geom['volume']
    
    spatter = predict_spatter_volume_from_PV(P, V)
    
    dw_ratio = depth / width if width > 0 else 0
    lw_ratio = length / width if width > 0 else 0
    
    defect_type = "Good"
    
    if dw_ratio > KEYHOLE_DW_THRESHOLD:
        defect_type = "Keyhole"
    elif depth < LOF_DEPTH_THRESHOLD:
        defect_type = "Lack of Fusion"
    elif BALLING_LW_MIN < lw_ratio < BALLING_LW_MAX:
        defect_type = "Balling"
    
    models = get_models()
    
    return {
        "material": material,
        "P": P,
        "V": V,
        "J": J,
        "spatter_volume": spatter,
        "length": length,
        "width": width,
        "depth": depth,
        "melt_pool_cross_section_um2": area,
        "melt_pool_volume_um3": volume,
        "defect_type": defect_type,
        "meta": {
            "input": {"P": P_in, "V": V_in, "J": J_in},
            "used": {"P": P_clamped, "V": V_clamped},
            "ranges": {"P": ranges["P"], "V": ranges["V"]},
            "clamped": {"P": P_in != P_clamped, "V": V_in != V_clamped},
            "model_source": {
                "depth": "poly_model" if 'Depth' in models else "fallback",
                "width": "poly_model" if 'Width' in models else "fallback",
                "length": "poly_model" if 'Length' in models else "fallback",
                "area": "poly_model" if 'Area' in models else "fallback",
                "volume": "poly_model" if 'Volume' in models else "fallback",
                "spatter": "poly_model" if 'Spatter' in models else "fallback"
            },
            "notes": {
                "ratios": {
                    "D/W": dw_ratio,
                    "L/W": lw_ratio,
                    "W/D": width / depth if depth > 0 else 0
                },
                "thresholds": {
                    "KEYHOLE_DW": KEYHOLE_DW_THRESHOLD,
                    "BALLING_LW_MIN": BALLING_LW_MIN,
                    "BALLING_LW_MAX": BALLING_LW_MAX,
                    "LOF_DEPTH": LOF_DEPTH_THRESHOLD
                }
            }
        }
    }


# =============================================================================
# HELPER: Compact result (strips meta for smaller JSON)
# =============================================================================

def compact_result(result: dict, precision: int = 2) -> dict:
    """
    Create a compact version of prediction result.
    Removes 'meta' field and rounds floats to reduce JSON size.
    """
    compact = {}
    for key, value in result.items():
        if key == "meta":
            continue
        elif isinstance(value, float):
            compact[key] = round(value, precision)
        else:
            compact[key] = value
    return compact


# =============================================================================
# INTERNAL IMPLEMENTATION FUNCTIONS (NOT DECORATED)
# =============================================================================
# These functions contain the actual logic and can be called from other
# functions without issues. The @mcp.tool() decorated functions delegate to these.

def _suggest_impl(
    material: str = "SS316L",
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    keep_P: Optional[float] = None,
    keep_V: Optional[float] = None,
    j_limits: ArrayParam = None,
    max_risk: float = 0.02,
    avoid_defects: bool = True,
    grid: int = 25,
    use_ranges: Optional[bool] = True,
    extra: Optional[dict] = None
) -> dict:
    """Core implementation for suggest logic."""
    P_bounds_parsed = parse_array_param(P_bounds)
    V_bounds_parsed = parse_array_param(V_bounds)
    j_limits_parsed = parse_array_param(j_limits)
    
    ranges = get_material_ranges(material)
    
    if j_limits_parsed:
        calculated_P, calculated_V = calculate_bounds_from_j_limits(
            j_limits=j_limits_parsed,
            keep_P=keep_P,
            keep_V=keep_V,
            current_P_bounds=P_bounds_parsed,
            current_V_bounds=V_bounds_parsed
        )
        if calculated_P is not None:
            P_bounds_parsed = calculated_P
        if calculated_V is not None:
            V_bounds_parsed = calculated_V
    
    if P_bounds_parsed:
        P_min, P_max = P_bounds_parsed[0], P_bounds_parsed[1]
    else:
        P_min, P_max = ranges["P"]
    
    if V_bounds_parsed:
        V_min, V_max = V_bounds_parsed[0], V_bounds_parsed[1]
    else:
        V_min, V_max = ranges["V"]
    
    P_min = max(P_min, ranges["P"][0])
    P_max = min(P_max, ranges["P"][1])
    V_min = max(V_min, ranges["V"][0])
    V_max = min(V_max, ranges["V"][1])
    
    if keep_P is not None:
        P_min = P_max = float(keep_P)
    if keep_V is not None:
        V_min = V_max = float(keep_V)
    
    P_grid = np.linspace(P_min, P_max, grid if P_min != P_max else 1)
    V_grid = np.linspace(V_min, V_max, grid if V_min != V_max else 1)
    
    points = []
    for P in P_grid:
        for V in V_grid:
            J = P / V
            
            if j_limits_parsed:
                if J < j_limits_parsed[0] or J > j_limits_parsed[1]:
                    continue
            
            result = predict_all_metrics(material, P, V)
            
            point = {
                "P": float(P),
                "V": float(V),
                "J": float(J),
                "spatter_volume": result["spatter_volume"],
                "defect_type": result["defect_type"]
            }
            points.append(point)
    
    if avoid_defects:
        safe_points = [p for p in points if p["defect_type"] == "Good"]
        used_safe = len(safe_points) > 0
        candidates = safe_points if safe_points else points
    else:
        candidates = points
        used_safe = False
    
    if candidates:
        best = min(candidates, key=lambda x: x["spatter_volume"])
    else:
        best = None
    
    return {
        "ok": True,
        "data": {
            "material": material,
            "used_ranges": {
                "P_bounds": [P_min, P_max],
                "V_bounds": [V_min, V_max],
                "j_limits": j_limits_parsed,
                "keep_P": keep_P,
                "keep_V": keep_V,
                "grid": grid,
                "max_risk": max_risk,
                "avoid_defects": avoid_defects
            },
            "points": points,
            "best": best,
            "used_safe_candidates": used_safe
        }
    }


def _filtered_grid_scan_impl(
    material: str = "SS316L",
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    j_limits: ArrayParam = None,
    grid_P: int = 21,
    grid_V: int = 21,
    max_spatter: Optional[float] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    min_length: Optional[float] = None,
    max_length: Optional[float] = None,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    max_risk: Optional[float] = None,
    defect_type: Optional[str] = None,
    top_n: Optional[int] = 10,
    sort_by: str = "spatter_volume",
    sort_desc: bool = False,
    extra: Optional[dict] = None
) -> dict:
    """
    Internal implementation for filtered grid scan.
    This is called by both mcp_filtered_grid_scan and mcp_find_best.
    """
    P_bounds_parsed = parse_array_param(P_bounds)
    V_bounds_parsed = parse_array_param(V_bounds)
    j_limits_parsed = parse_array_param(j_limits)
    
    ranges = get_material_ranges(material)
    
    if j_limits_parsed:
        calculated_P, calculated_V = calculate_bounds_from_j_limits(
            j_limits=j_limits_parsed,
            current_P_bounds=P_bounds_parsed,
            current_V_bounds=V_bounds_parsed
        )
        if calculated_P is not None and P_bounds_parsed is None:
            P_bounds_parsed = calculated_P
        if calculated_V is not None and V_bounds_parsed is None:
            V_bounds_parsed = calculated_V
    
    if P_bounds_parsed:
        P_min, P_max = P_bounds_parsed
    else:
        P_min, P_max = ranges["P"]
    
    if V_bounds_parsed:
        V_min, V_max = V_bounds_parsed
    else:
        V_min, V_max = ranges["V"]
    
    P_min = max(P_min, ranges["P"][0])
    P_max = min(P_max, ranges["P"][1])
    V_min = max(V_min, ranges["V"][0])
    V_max = min(V_max, ranges["V"][1])
    
    P_grid = np.linspace(P_min, P_max, grid_P)
    V_grid = np.linspace(V_min, V_max, grid_V)
    
    total_scanned = 0
    filtered_results = []
    
    for P in P_grid:
        for V in V_grid:
            J = P / V
            
            if j_limits_parsed:
                if J < j_limits_parsed[0] or J > j_limits_parsed[1]:
                    continue
            
            total_scanned += 1
            result = predict_all_metrics(material, float(P), float(V))
            
            # Apply filters
            if max_spatter is not None and result["spatter_volume"] > max_spatter:
                continue
            if min_depth is not None and result["depth"] < min_depth:
                continue
            if max_depth is not None and result["depth"] > max_depth:
                continue
            if min_length is not None and result["length"] < min_length:
                continue
            if max_length is not None and result["length"] > max_length:
                continue
            if min_width is not None and result["width"] < min_width:
                continue
            if max_width is not None and result["width"] > max_width:
                continue
            if defect_type is not None and result["defect_type"] != defect_type:
                continue
            
            filtered_results.append(compact_result(result))
    
    if filtered_results and sort_by:
        filtered_results.sort(
            key=lambda x: x.get(sort_by, 0),
            reverse=sort_desc
        )
    
    if top_n is not None and len(filtered_results) > top_n:
        filtered_results = filtered_results[:top_n]
    
    filters_applied = {}
    if max_spatter is not None:
        filters_applied["max_spatter"] = max_spatter
    if min_depth is not None or max_depth is not None:
        filters_applied["depth"] = [min_depth, max_depth]
    if min_length is not None or max_length is not None:
        filters_applied["length"] = [min_length, max_length]
    if min_width is not None or max_width is not None:
        filters_applied["width"] = [min_width, max_width]
    if defect_type is not None:
        filters_applied["defect_type"] = defect_type
    
    return {
        "ok": True,
        "data": {
            "material": material,
            "grid": f"{grid_P}x{grid_V}",
            "total_scanned": total_scanned,
            "matches_found": len(filtered_results),
            "filters_applied": filters_applied,
            "sort_by": sort_by,
            "best": filtered_results[0] if filtered_results else None,
            "results": filtered_results
        }
    }


def _find_best_impl(
    material: str = "SS316L",
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    j_limits: ArrayParam = None,
    max_spatter: Optional[float] = None,
    depth_range: ArrayParam = None,
    length_range: ArrayParam = None,
    width_range: ArrayParam = None,
    max_risk: float = 0.02,
    defect_free: bool = True,
    minimize: str = "spatter_volume",
    grid: int = 25,
    extra: Optional[dict] = None
) -> dict:
    """
    Internal implementation for find_best.
    This is called by both mcp_find_best and mcp_optimize.
    """
    depth_range_parsed = parse_array_param(depth_range)
    length_range_parsed = parse_array_param(length_range)
    width_range_parsed = parse_array_param(width_range)
    
    min_depth = depth_range_parsed[0] if depth_range_parsed and depth_range_parsed[0] is not None else None
    max_depth = depth_range_parsed[1] if depth_range_parsed and len(depth_range_parsed) > 1 and depth_range_parsed[1] is not None else None
    min_length = length_range_parsed[0] if length_range_parsed and length_range_parsed[0] is not None else None
    max_length = length_range_parsed[1] if length_range_parsed and len(length_range_parsed) > 1 and length_range_parsed[1] is not None else None
    min_width = width_range_parsed[0] if width_range_parsed and width_range_parsed[0] is not None else None
    max_width = width_range_parsed[1] if width_range_parsed and len(width_range_parsed) > 1 and width_range_parsed[1] is not None else None
    
    # Call internal implementation instead of decorated tool
    result = _filtered_grid_scan_impl(
        material=material,
        P_bounds=P_bounds,
        V_bounds=V_bounds,
        j_limits=j_limits,
        grid_P=grid,
        grid_V=grid,
        max_spatter=max_spatter,
        min_depth=min_depth,
        max_depth=max_depth,
        min_length=min_length,
        max_length=max_length,
        min_width=min_width,
        max_width=max_width,
        defect_type="Good" if defect_free else None,
        top_n=5,
        sort_by=minimize,
        sort_desc=False
    )
    
    if not result["ok"]:
        return result
    
    data = result["data"]
    best = data.get("best")
    all_results = data.get("results", [])
    
    return {
        "ok": True,
        "data": {
            "status": "found" if best else "no_match",
            "material": material,
            "best": best,
            "alternatives": all_results[1:5] if len(all_results) > 1 else [],
            "search_summary": {
                "total_scanned": data["total_scanned"],
                "matches_found": data["matches_found"],
                "grid": data["grid"],
                "constraints": data["filters_applied"],
                "minimize": minimize
            }
        }
    }


# =============================================================================
# MARGIN-AWARE (ROBUST) OPTIMIZATION
# =============================================================================

def threshold_margins(P: float, V: float, depth: float, width: float, length: float, ranges: dict) -> dict:
    """Distance of a point to every regime threshold and to the edge of the trained range.

    Positive values mean the point is on the Good side of that threshold.
      depth_um    : D - LOF_DEPTH_THRESHOLD                      (um)
      dw          : KEYHOLE_DW_THRESHOLD - D/W                   (-)
      lw          : distance of L/W outside the balling band     (-)
      range       : smallest distance of P or V to the trained-range edge, as a fraction of the range width
    'nearest' names the threshold with the smallest relative margin (margin / threshold value).
    """
    dw = depth / width if width > 0 else float("inf")
    lw = length / width if width > 0 else 0.0
    m_depth = depth - LOF_DEPTH_THRESHOLD
    m_dw = KEYHOLE_DW_THRESHOLD - dw
    if lw <= BALLING_LW_MIN:
        m_lw = BALLING_LW_MIN - lw
    elif lw >= BALLING_LW_MAX:
        m_lw = lw - BALLING_LW_MAX
    else:
        m_lw = -min(lw - BALLING_LW_MIN, BALLING_LW_MAX - lw)          # inside the band
    (p0, p1), (v0, v1) = ranges["P"], ranges["V"]
    m_range = min((P - p0) / (p1 - p0), (p1 - P) / (p1 - p0), (V - v0) / (v1 - v0), (v1 - V) / (v1 - v0))
    rel = {"lack_of_fusion": m_depth / LOF_DEPTH_THRESHOLD, "keyhole": m_dw / KEYHOLE_DW_THRESHOLD,
           "balling": m_lw / BALLING_LW_MIN}
    nearest = min(rel, key=rel.get)
    return {"depth_um": m_depth, "dw": m_dw, "lw": m_lw, "range": m_range,
            "nearest_threshold": nearest, "nearest_relative_margin": rel[nearest]}


def _find_robust_impl(material: str = "SS316L", P_bounds: ArrayParam = None, V_bounds: ArrayParam = None,
                      j_limits: ArrayParam = None, max_spatter: Optional[float] = None,
                      depth_range: ArrayParam = None, length_range: ArrayParam = None, width_range: ArrayParam = None,
                      min_depth_margin: float = 15.0, min_dw_margin: float = 0.10, min_lw_margin: float = 0.15,
                      min_range_margin: float = 0.05, grid: int = 46, top_n: int = 5) -> dict:
    ranges = get_material_ranges(material)
    Pb = parse_array_param(P_bounds) or ranges["P"]; Vb = parse_array_param(V_bounds) or ranges["V"]
    Pb = [max(ranges["P"][0], Pb[0]), min(ranges["P"][1], Pb[1])]; Vb = [max(ranges["V"][0], Vb[0]), min(ranges["V"][1], Vb[1])]
    J = parse_array_param(j_limits); dr = parse_array_param(depth_range); lr = parse_array_param(length_range); wr = parse_array_param(width_range)
    inside = lambda x, r: r is None or ((r[0] is None or x >= r[0]) and (len(r) < 2 or r[1] is None or x <= r[1]))
    grid = int(max(2, min(grid, 80)))
    Ps = [Pb[0] + i * (Pb[1] - Pb[0]) / (grid - 1) for i in range(grid)] if Pb[1] > Pb[0] else [Pb[0]]
    Vs = [Vb[0] + i * (Vb[1] - Vb[0]) / (grid - 1) for i in range(grid)] if Vb[1] > Vb[0] else [Vb[0]]
    plain, robust = [], []
    for P in Ps:
        for V in Vs:
            r = predict_all_metrics(material, P, V)
            if r["defect_type"] != "Good": continue
            if J and not inside(r["J"], J): continue
            if max_spatter is not None and r["spatter_volume"] > max_spatter: continue
            if not (inside(r["depth"], dr) and inside(r["length"], lr) and inside(r["width"], wr)): continue
            m = threshold_margins(P, V, r["depth"], r["width"], r["length"], ranges)
            pt = {"P": round(P, 1), "V": round(V, 1), "J": round(r["J"], 3), "spatter_volume": round(r["spatter_volume"], 1),
                  "depth": round(r["depth"], 1), "width": round(r["width"], 1), "length": round(r["length"], 1),
                  "defect_type": "Good", "margins": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}}
            plain.append(pt)
            if m["depth_um"] >= min_depth_margin and m["dw"] >= min_dw_margin and m["lw"] >= min_lw_margin and m["range"] >= min_range_margin:
                robust.append(pt)
    plain.sort(key=lambda p: p["spatter_volume"]); robust.sort(key=lambda p: p["spatter_volume"])
    best, ref = (robust[0] if robust else None), (plain[0] if plain else None)
    cost = None
    if best and ref and ref["spatter_volume"] > 0:
        cost = round(100 * (best["spatter_volume"] - ref["spatter_volume"]) / ref["spatter_volume"], 1)
    return {"ok": True, "data": {
        "status": "found" if best else ("no_robust_point" if plain else "no_match"),
        "material": material, "best": best, "alternatives": robust[1:top_n],
        "minimum_spatter_without_margins": ref,
        "spatter_cost_of_margins_pct": cost,
        "margins_required": {"depth_um": min_depth_margin, "dw": min_dw_margin, "lw": min_lw_margin, "range_fraction": min_range_margin},
        "search_summary": {"grid": f"{len(Ps)}x{len(Vs)}", "good_points_meeting_constraints": len(plain), "points_meeting_margins": len(robust)},
        "note": ("No defect-free point keeps the requested margins; relax a margin or a constraint." if plain and not robust else None)}}


# =============================================================================
# MCP TOOL ENDPOINTS - MATERIAL INFO
# =============================================================================

@mcp.tool()
def mcp_list_materials() -> dict:
    """
    List all available materials in the dataset.
    
    Call this first to see what materials are supported.
    
    Returns:
        Dictionary with:
        - materials: List of available material names
        - count: Number of materials
        - default: Default material if none specified
    """
    return {
        "ok": True,
        "data": {
            "materials": list(MATERIAL_BOUNDS.keys()),
            "count": len(MATERIAL_BOUNDS),
            "default": DEFAULT_MATERIAL
        }
    }


@mcp.tool()
def mcp_get_material_info(material: str) -> dict:
    """
    Get valid parameter ranges for a material based on the training dataset.
    
    IMPORTANT: Call this FIRST before running any searches or optimizations 
    to ensure you're searching within the valid data range.
    
    Args:
        material: Alloy name (only "SS316L" is supported)
    
    Returns:
        Dictionary containing:
        - P_range: [min, max] power in Watts
        - V_range: [min, max] velocity in mm/s
        - J_range: [min, max] energy density in J/mm
        - P_typical: [10th, 90th percentile] - where most data lives
        - V_typical: [10th, 90th percentile] - where most data lives
        - data_points: number of training samples
        - source: data source description
        - units: dictionary of units for each parameter
    
    Example:
        info = mcp_get_material_info("SS316L")
        # Returns P_range: [160, 430], V_range: [100, 1500]
        # Use these bounds in mcp_filtered_grid_scan or mcp_find_best
    """
    material_key = None
    for m in MATERIAL_BOUNDS.keys():
        if m.lower() == material.lower():
            material_key = m
            break
    
    if material_key is None:
        return {
            "ok": False,
            "error": f"Unknown material: '{material}'",
            "available_materials": list(MATERIAL_BOUNDS.keys()),
            "suggestion": "Use mcp_list_materials() to see all available materials"
        }
    
    info = MATERIAL_BOUNDS[material_key].copy()
    info["material"] = material_key
    
    return {
        "ok": True,
        "data": info
    }


@mcp.tool()
def mcp_get_all_materials_info() -> dict:
    """
    Get parameter ranges for ALL materials in one call.
    
    Useful for comparing across materials or building UIs.
    
    Returns:
        Dictionary with all materials and their bounds, including:
        - P_range, V_range, J_range for each material
        - data_points count
        - units
    """
    return {
        "ok": True,
        "data": {
            "materials": MATERIAL_BOUNDS,
            "count": len(MATERIAL_BOUNDS),
            "default": DEFAULT_MATERIAL
        }
    }


# =============================================================================
# MCP TOOL ENDPOINTS - PREDICTION
# =============================================================================

@mcp.tool()
def ping(msg: str = "pong") -> dict:
    """Lightweight liveness probe."""
    return {"ok": True, "msg": msg}


@mcp.tool()
def mcp_predict(
    material: str,
    P: float,
    V: float,
    extra: Optional[dict] = None
) -> dict:
    """Predict a single target or small set of metrics for (material, P, V)."""
    result = predict_spatter(material, P, V, extra)
    return {"ok": True, "data": result}


@mcp.tool()
def mcp_predict_all(
    material: str,
    P: float,
    V: float,
    extra: Optional[dict] = None
) -> dict:
    """Predict spatter/defect metrics for given material and (P, V)."""
    result = predict_all_metrics(material, P, V, extra)
    return {"ok": True, "data": result}


@mcp.tool()
def mcp_predict_spatter(
    material: str,
    P: float,
    V: float,
    extra: Optional[dict] = None
) -> dict:
    """Predict spatter volume only."""
    result = predict_spatter(material, P, V, extra)
    return {"ok": True, "data": {"spatter_volume": result["spatter_volume"], "P_used": result["P"],
                                 "V_used": result["V"], "clamped": result["meta"]["clamped"],
                                 "trained_range": result["meta"]["ranges"]}}


@mcp.tool()
def mcp_predict_defect(
    material: str,
    P: float,
    V: float,
    extra: Optional[dict] = None
) -> dict:
    """Predict defect type."""
    result = predict_all_metrics(material, P, V, extra)
    return {"ok": True, "data": {
        "defect_type": result["defect_type"],
        "P_used": result["P"], "V_used": result["V"],
        "clamped": result["meta"]["clamped"],
        "trained_range": result["meta"]["ranges"],
    }}


# =============================================================================
# MCP TOOL ENDPOINTS - OPTIMIZATION
# =============================================================================

@mcp.tool()
def mcp_suggest(
    material: str = "SS316L",
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    keep_P: Optional[float] = None,
    keep_V: Optional[float] = None,
    j_limits: ArrayParam = None,
    max_risk: float = 0.02,
    avoid_defects: bool = True,
    grid: int = 25,
    use_ranges: Optional[bool] = True,
    extra: Optional[dict] = None
) -> dict:
    """Suggest P,V to minimize spatter+risk; supports locks, bounds, J-window, and risk cap."""
    return _suggest_impl(
        material=material,
        P_bounds=P_bounds,
        V_bounds=V_bounds,
        keep_P=keep_P,
        keep_V=keep_V,
        j_limits=j_limits,
        max_risk=max_risk,
        avoid_defects=avoid_defects,
        grid=grid,
        use_ranges=use_ranges,
        extra=extra
    )


@mcp.tool()
def mcp_suggest_next(
    material: str,
    n: int = 1,
    extra: Optional[dict] = None
) -> dict:
    """Suggest immediate next best point(s)."""
    result = _suggest_impl(material=material, grid=20)
    
    if result["ok"] and result["data"]["points"]:
        sorted_points = sorted(
            [p for p in result["data"]["points"] if p["defect_type"] == "Good"],
            key=lambda x: x["spatter_volume"]
        )[:n]
        return {"ok": True, "data": {"suggestions": sorted_points}}
    
    return {"ok": False, "error": "No valid points found"}


@mcp.tool()
def mcp_line_scan_min_spatter(
    material: str = "SS316L",
    keep_P: Optional[float] = None,
    keep_V: Optional[float] = None,
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    j_limits: ArrayParam = None,
    max_risk: float = 0.02,
    grid: int = 40,
    n: Optional[int] = None,
    p_min: Optional[float] = None,
    p_max: Optional[float] = None,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
    extra: Optional[dict] = None
) -> dict:
    """
    Min-spatter 1-D scan. Lock exactly one of keep_P or keep_V.
    
    Now properly handles j_limits:
    - If keep_P is set with j_limits, V_bounds are auto-calculated
    - If keep_V is set with j_limits, P_bounds are auto-calculated
    """
    P_bounds_parsed = parse_array_param(P_bounds)
    V_bounds_parsed = parse_array_param(V_bounds)
    j_limits_parsed = parse_array_param(j_limits)
    
    if p_min is not None and p_max is not None:
        P_bounds_parsed = [p_min, p_max]
    if v_min is not None and v_max is not None:
        V_bounds_parsed = [v_min, v_max]
    
    return _suggest_impl(
        material=material,
        P_bounds=P_bounds_parsed,
        V_bounds=V_bounds_parsed,
        keep_P=keep_P,
        keep_V=keep_V,
        j_limits=j_limits_parsed,
        max_risk=max_risk,
        grid=grid,
        extra=extra
    )


# =============================================================================
# MCP TOOL ENDPOINTS - SCANNING
# =============================================================================

@mcp.tool()
def mcp_grid_scan(
    material: str = "SS316L",
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    j_limits: ArrayParam = None,
    grid_P: int = 21,
    grid_V: int = 21,
    compact: bool = False,
    extra: Optional[dict] = None
) -> dict:
    """
    Exhaustive 2D grid scan over (P, V) using predict_all.
    
    Now supports j_limits to constrain the energy density range.
    Points outside j_limits are filtered out.
    
    Args:
        material: Alloy name (default: SS316L)
        P_bounds: [P_min, P_max] power range in Watts
        V_bounds: [V_min, V_max] velocity range in mm/s
        j_limits: [J_min, J_max] energy density range
        grid_P: Number of power grid points (default: 21)
        grid_V: Number of velocity grid points (default: 21)
        compact: If True, strip 'meta' field to reduce JSON size (~60% smaller)
        extra: Additional parameters (unused)
    
    Returns:
        Grid scan results with all predictions
    
    Note: For large grids (>15x15), use compact=True to avoid JSON size issues.
    For filtered results, use mcp_filtered_grid_scan instead.
    """
    P_bounds_parsed = parse_array_param(P_bounds)
    V_bounds_parsed = parse_array_param(V_bounds)
    j_limits_parsed = parse_array_param(j_limits)
    
    ranges = get_material_ranges(material)
    
    if j_limits_parsed:
        calculated_P, calculated_V = calculate_bounds_from_j_limits(
            j_limits=j_limits_parsed,
            current_P_bounds=P_bounds_parsed,
            current_V_bounds=V_bounds_parsed
        )
        if calculated_P is not None and P_bounds_parsed is None:
            P_bounds_parsed = calculated_P
        if calculated_V is not None and V_bounds_parsed is None:
            V_bounds_parsed = calculated_V
    
    if P_bounds_parsed:
        P_min, P_max = P_bounds_parsed
    else:
        P_min, P_max = ranges["P"]
    
    if V_bounds_parsed:
        V_min, V_max = V_bounds_parsed
    else:
        V_min, V_max = ranges["V"]
    
    P_min = max(P_min, ranges["P"][0])
    P_max = min(P_max, ranges["P"][1])
    V_min = max(V_min, ranges["V"][0])
    V_max = min(V_max, ranges["V"][1])
    
    P_grid = np.linspace(P_min, P_max, grid_P)
    V_grid = np.linspace(V_min, V_max, grid_V)
    
    results = []
    for P in P_grid:
        for V in V_grid:
            J = P / V
            
            if j_limits_parsed:
                if J < j_limits_parsed[0] or J > j_limits_parsed[1]:
                    continue
            
            result = predict_all_metrics(material, float(P), float(V))
            
            if compact:
                result = compact_result(result)
            
            results.append(result)
    
    return {
        "ok": True,
        "data": {
            "material": material,
            "P_bounds": [P_min, P_max],
            "V_bounds": [V_min, V_max],
            "j_limits": j_limits_parsed,
            "grid_P": grid_P,
            "grid_V": grid_V,
            "compact": compact,
            "total_points": len(results),
            "results": results
        }
    }


@mcp.tool()
def mcp_filtered_grid_scan(
    material: str = "SS316L",
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    j_limits: ArrayParam = None,
    grid_P: int = 21,
    grid_V: int = 21,
    max_spatter: Optional[float] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    min_length: Optional[float] = None,
    max_length: Optional[float] = None,
    min_width: Optional[float] = None,
    max_width: Optional[float] = None,
    max_risk: Optional[float] = None,
    defect_type: Optional[str] = None,
    top_n: Optional[int] = 10,
    sort_by: str = "spatter_volume",
    sort_desc: bool = False,
    extra: Optional[dict] = None
) -> dict:
    """
    Grid scan with SERVER-SIDE filtering - returns only matching points.
    
    This solves the large JSON problem by filtering BEFORE returning results,
    so you only get the points you care about.
    
    Args:
        material: Alloy name (default: SS316L)
        P_bounds: [P_min, P_max] power range in Watts
        V_bounds: [V_min, V_max] velocity range in mm/s
        j_limits: [J_min, J_max] energy density range
        grid_P: Number of power grid points (default: 21)
        grid_V: Number of velocity grid points (default: 21)
        
        FILTERS (all optional - only applied if set):
        max_spatter: Maximum spatter volume (µm³)
        min_depth: Minimum melt pool depth (µm)
        max_depth: Maximum melt pool depth (µm)
        min_length: Minimum melt pool length (µm)
        max_length: Maximum melt pool length (µm)
        min_width: Minimum melt pool width (µm)
        max_width: Maximum melt pool width (µm)
        max_risk: Maximum defect probability (0-1) - NOT USED (kept for API compatibility)
        defect_type: Filter by defect type ("Good", "Keyhole", "Lack of Fusion", "Balling")
        
        OUTPUT CONTROL:
        top_n: Return only top N results (default: 10, set None for all)
        sort_by: Field to sort by (default: "spatter_volume")
        sort_desc: Sort descending if True (default: False = ascending)
    
    Returns:
        Filtered results with best point highlighted
    
    Example - find best point with spatter < 80000, depth 105-130, length < 500:
        mcp_filtered_grid_scan(
            material="SS316L",
            max_spatter=80000,
            min_depth=105,
            max_depth=130,
            max_length=500,
            top_n=5
        )
    """
    # Delegate to internal implementation
    return _filtered_grid_scan_impl(
        material=material,
        P_bounds=P_bounds,
        V_bounds=V_bounds,
        j_limits=j_limits,
        grid_P=grid_P,
        grid_V=grid_V,
        max_spatter=max_spatter,
        min_depth=min_depth,
        max_depth=max_depth,
        min_length=min_length,
        max_length=max_length,
        min_width=min_width,
        max_width=max_width,
        max_risk=max_risk,
        defect_type=defect_type,
        top_n=top_n,
        sort_by=sort_by,
        sort_desc=sort_desc,
        extra=extra
    )


@mcp.tool()
def mcp_find_best(
    material: str = "SS316L",
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    j_limits: ArrayParam = None,
    max_spatter: Optional[float] = None,
    depth_range: ArrayParam = None,
    length_range: ArrayParam = None,
    width_range: ArrayParam = None,
    max_risk: float = 0.02,
    defect_free: bool = True,
    minimize: str = "spatter_volume",
    grid: int = 25,
    extra: Optional[dict] = None
) -> dict:
    """
    Find the single best (P, V) point matching all constraints.
    
    A convenience function that returns just the optimal point and a few
    alternatives, rather than a full grid of results.
    
    Args:
        material: Alloy name (default: SS316L)
        P_bounds: [P_min, P_max] power range in Watts
        V_bounds: [V_min, V_max] velocity range in mm/s
        j_limits: [J_min, J_max] energy density range
        
        CONSTRAINTS:
        max_spatter: Maximum allowed spatter volume (µm³)
        depth_range: [min_depth, max_depth] in µm
        length_range: [min_length, max_length] in µm
        width_range: [min_width, max_width] in µm
        max_risk: (kept for API compatibility, not used)
        defect_free: If True, only consider "Good" defect type (default: True)
        
        OPTIMIZATION:
        minimize: Metric to minimize (default: "spatter_volume")
        grid: Grid resolution (default: 25)
    
    Returns:
        Best point matching all constraints, plus up to 4 alternatives
    
    Example:
        mcp_find_best(
            material="SS316L",
            max_spatter=80000,
            depth_range=[105, 130],
            length_range=[None, 500]
        )
    """
    # Delegate to internal implementation
    return _find_best_impl(
        material=material,
        P_bounds=P_bounds,
        V_bounds=V_bounds,
        j_limits=j_limits,
        max_spatter=max_spatter,
        depth_range=depth_range,
        length_range=length_range,
        width_range=width_range,
        max_risk=max_risk,
        defect_free=defect_free,
        minimize=minimize,
        grid=grid,
        extra=extra
    )


@mcp.tool()
def mcp_find_robust(
    material: str = "SS316L",
    P_bounds: ArrayParam = None,
    V_bounds: ArrayParam = None,
    j_limits: ArrayParam = None,
    max_spatter: Optional[float] = None,
    depth_range: ArrayParam = None,
    length_range: ArrayParam = None,
    width_range: ArrayParam = None,
    min_depth_margin: float = 15.0,
    min_dw_margin: float = 0.10,
    min_lw_margin: float = 0.15,
    min_range_margin: float = 0.05,
    grid: int = 46,
    top_n: int = 5,
) -> dict:
    """
    Margin-aware optimization: lowest-spatter defect-free (P, V) that also stays a safe distance
    from every regime threshold and from the edge of the trained range.

    Use this when the user asks for robust, safe or production parameters, a safety margin,
    or parameters that tolerate machine drift. Plain minimum-spatter optima tend to sit right at the
    lack-of-fusion boundary or at the edge of the trained range; this tool avoids that.

    Margins (all must hold):
        min_depth_margin : D - 137 um  >= this value (um), distance from lack of fusion
        min_dw_margin    : 1.25 - D/W  >= this value, distance from keyhole
        min_lw_margin    : distance of L/W outside the balling band 2.35-3.7 >= this value
        min_range_margin : distance of P and V from the trained-range edge, as a fraction of the range (0-0.5)
    Other constraints (P_bounds, V_bounds, j_limits, max_spatter, depth/length/width ranges) as in mcp_find_best.

    Returns the best robust point and alternatives with their margins and nearest threshold, the plain
    minimum-spatter point for comparison, and the spatter cost of the margins in percent.
    """
    return _find_robust_impl(material, P_bounds, V_bounds, j_limits, max_spatter, depth_range, length_range,
                             width_range, min_depth_margin, min_dw_margin, min_lw_margin, min_range_margin, grid, top_n)


@mcp.tool()
def mcp_optimize(
    material: str,
    objective: str = "min_spatter",
    constraints: Optional[dict] = None,
    extra: Optional[dict] = None
) -> dict:
    """
    Global optimization over (P,V) for an objective with constraints.
    
    Constraints can include:
    - j_limits: [J_min, J_max] - energy density range
    - P_bounds: [P_min, P_max] - power range
    - V_bounds: [V_min, V_max] - velocity range
    - max_spatter: float - maximum spatter volume
    - depth_range: [min, max] - depth constraints
    - length_range: [min, max] - length constraints
    """
    j_limits = None
    P_bounds = None
    V_bounds = None
    max_spatter = None
    depth_range = None
    length_range = None
    
    if constraints:
        j_limits = constraints.get("j_limits")
        P_bounds = constraints.get("P_bounds")
        V_bounds = constraints.get("V_bounds")
        max_spatter = constraints.get("max_spatter")
        depth_range = constraints.get("depth_range")
        length_range = constraints.get("length_range")
    
    # Delegate to internal implementation
    result = _find_best_impl(
        material=material,
        P_bounds=P_bounds,
        V_bounds=V_bounds,
        j_limits=j_limits,
        max_spatter=max_spatter,
        depth_range=depth_range,
        length_range=length_range,
        defect_free=True,
        minimize="spatter_volume",
        grid=30
    )
    
    if result["ok"] and result["data"]["best"]:
        return {
            "ok": True,
            "data": {
                "material": material,
                "objective": objective,
                "constraints": constraints,
                "optimal": result["data"]["best"],
                "alternatives": result["data"]["alternatives"],
                "method": "filtered_grid_search"
            }
        }
    
    return {"ok": False, "error": "Optimization failed - no valid points found"}


# =============================================================================
# MCP TOOL ENDPOINTS - EXPERIMENT MANAGEMENT
# =============================================================================

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Pre-load models at startup
    load_models()
    mcp.run()
