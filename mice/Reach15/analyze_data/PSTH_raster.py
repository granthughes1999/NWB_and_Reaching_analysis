from __future__ import annotations

"""Shared PSTH/raster helpers extracted from 03_psth_raster_NWB_testing.ipynb."""

import gc
from colorsys import hls_to_rgb, rgb_to_hls
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - optional dependency
    gaussian_filter1d = None


DEFAULT_SPLIT_COLORS = {
    'baseline': 'green',
    'optical_stim': 'blue',
    'no_optical_stim': 'red',
}

DEFAULT_REGION_BACKGROUND_COLORS = [
    '#d8e7f2',
    '#d6e8cf',
    '#ead6ef',
    '#f0d8cd',
    '#d2e7e0',
    '#f0d6dc',
    '#ddd6b2',
    '#d9dff2',
]

SPECIAL_UNSPLIT_EVENTS = {
    'first_opto_tagging_timestamp_per_trial',
    'first_optical_pulse_per_closed_loop',
}


def _coerce_window(pre, post):
    if pre is None or post is None:
        raise ValueError('pre and post are required.')
    pre = abs(float(pre))
    post = abs(float(post))
    if pre == 0 or post == 0:
        raise ValueError('pre and post must be non-zero.')
    return pre, post


def flatten_nested_trial_numbers(nested):
    if nested is None:
        return np.array([], dtype=int)
    if isinstance(nested, np.ndarray) and nested.dtype != object:
        return nested.astype(int).ravel()
    if isinstance(nested, (list, tuple, np.ndarray)):
        parts = [flatten_nested_trial_numbers(item) for item in nested]
        parts = [part for part in parts if part.size > 0]
        if not parts:
            return np.array([], dtype=int)
        return np.concatenate(parts).astype(int)
    return np.array([int(nested)], dtype=int)


def build_trial_index_groups(
    baseline_trials_idx,
    optoicalStim_trials_idx,
    washout_trials_idx,
    *,
    one_based=True,
):
    offset = 1 if one_based else 0
    out = {
        'baseline': flatten_nested_trial_numbers(baseline_trials_idx) - offset,
        'optical_stim': flatten_nested_trial_numbers(optoicalStim_trials_idx) - offset,
        'no_optical_stim': flatten_nested_trial_numbers(washout_trials_idx) - offset,
    }
    return {key: values[values >= 0] for key, values in out.items()}


def combine_merged_units(merged_dic):
    frames = []
    for probe, frame in merged_dic.items():
        piece = frame.copy()
        if 'probe' not in piece.columns:
            piece['probe'] = str(probe)
        frames.append(piece)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _cluster_id_series(df_units):
    if 'cluster_id' in df_units.columns:
        return pd.to_numeric(df_units['cluster_id'], errors='coerce')
    return pd.Series(df_units.index, index=df_units.index, dtype=float)


def _region_column(df_units):
    for name in ['Brain_Region', 'brain_region']:
        if name in df_units.columns:
            return name
    return None


def _normalize_region_value(value):
    if pd.isna(value):
        return 'unknown_region'
    text = str(value).strip()
    if text in {'', 'None', 'nan'}:
        return 'unknown_region'
    return text


def _unit_label_column(df_units, *, use_label=False, use_kslabel=False, kslabel_value=None):
    if use_label and 'label' in df_units.columns:
        return 'label', 2
    if use_kslabel and 'KSlabel' in df_units.columns:
        return 'KSlabel', 2 if kslabel_value is None else kslabel_value
    if kslabel_value is not None and 'KSlabel' in df_units.columns:
        return 'KSlabel', kslabel_value
    return None, None


def _bombcell_good_mask(df_units):
    if 'bc_unitType' in df_units.columns:
        values = df_units['bc_unitType'].fillna('').astype(str).str.strip().str.upper()
        reject = values.isin({'', 'NONE', 'NAN', 'NOISE', 'NON-SOMA', 'NON SOMA'})
        accept = values.eq('SOMA') | values.str.contains('GOOD', na=False)
        if accept.any():
            return accept & ~reject
        return ~reject

    if 'bc_label' in df_units.columns:
        values = df_units['bc_label'].fillna('').astype(str).str.strip().str.upper()
        reject = (
            values.eq('')
            | values.eq('NONE')
            | values.eq('NAN')
            | values.str.startswith('NOISE')
            | values.str.startswith('NON-SOMA')
            | values.str.startswith('NON SOMA')
        )
        accept = values.str.startswith('SOMA') | values.str.startswith('GOOD')
        if accept.any():
            return accept & ~reject
        return ~reject

    raise ValueError('bombcell_label=True requires bc_unitType or bc_label column.')


def _select_units(
    df_units,
    *,
    probeLetter=None,
    brain_region=None,
    label=False,
    KSlabel=False,
    bombcell_label=False,
    KSlabel_good=False,
    kslabel_mua=False,
    all_units=True,
):
    df1 = df_units.copy()
    if probeLetter is not None and 'probe' in df1.columns:
        df1 = df1[df1['probe'].astype(str) == str(probeLetter)]

    region_col = _region_column(df1)
    if brain_region is not None and region_col is not None:
        df1 = df1[df1[region_col].astype(str) == str(brain_region)]

    filter_flags = [bool(label), bool(KSlabel), bool(bombcell_label), bool(KSlabel_good), bool(kslabel_mua)]
    if sum(filter_flags) > 1:
        raise ValueError('Use only one unit-quality filter at a time.')

    col = None
    target = None
    if label:
        col, target = _unit_label_column(df1, use_label=True)
    elif KSlabel or KSlabel_good:
        col, target = _unit_label_column(df1, use_kslabel=True, kslabel_value=2)
    elif kslabel_mua:
        col, target = _unit_label_column(df1, kslabel_value=1)

    if col is not None:
        df1 = df1[pd.to_numeric(df1[col], errors='coerce') == target]
    elif bombcell_label:
        df1 = df1[_bombcell_good_mask(df1)]

    if not all_units and col is None:
        df1 = df1.copy()

    return df1.reset_index(drop=True)


def select_units(
    df_units,
    *,
    probeLetter=None,
    brain_region=None,
    label=False,
    KSlabel=False,
    bombcell_label=False,
    KSlabel_good=False,
    kslabel_mua=False,
    all_units=True,
):
    return _select_units(
        df_units,
        probeLetter=probeLetter,
        brain_region=brain_region,
        label=label,
        KSlabel=KSlabel,
        bombcell_label=bombcell_label,
        KSlabel_good=KSlabel_good,
        kslabel_mua=kslabel_mua,
        all_units=all_units,
    )


def split_units_by_probe_and_region(
    df_units,
    *,
    probes=None,
    label=False,
    KSlabel=False,
    bombcell_label=False,
    KSlabel_good=False,
    kslabel_mua=False,
    include_unknown=True,
):
    df1 = _select_units(
        df_units,
        label=label,
        KSlabel=KSlabel,
        bombcell_label=bombcell_label,
        KSlabel_good=KSlabel_good,
        kslabel_mua=kslabel_mua,
    )
    if df1.empty:
        return []

    if probes is not None and 'probe' in df1.columns:
        probe_set = {str(probe) for probe in probes}
        df1 = df1[df1['probe'].astype(str).isin(probe_set)].copy()
    if df1.empty:
        return []

    region_col = _region_column(df1)
    grouped = df1.copy()
    if region_col is None:
        grouped['_plot_region'] = 'unknown_region'
    else:
        grouped['_plot_region'] = grouped[region_col].map(_normalize_region_value)

    if 'probe' in grouped.columns:
        probe_values = grouped['probe'].fillna('unknown_probe').astype(str)
    else:
        probe_values = pd.Series(['unknown_probe'] * len(grouped), index=grouped.index, dtype='object')
    grouped['_plot_probe'] = probe_values

    out = []
    for probe in sorted(grouped['_plot_probe'].dropna().astype(str).unique().tolist()):
        probe_df = grouped[grouped['_plot_probe'] == probe].copy()
        region_labels = probe_df['_plot_region'].dropna().astype(str).unique().tolist()
        for region_label in region_labels:
            if region_label == 'unknown_region' and not include_unknown:
                continue
            region_df = probe_df[probe_df['_plot_region'] == region_label].copy()
            region_df = region_df.drop(columns=['_plot_region', '_plot_probe']).reset_index(drop=True)
            out.append({
                'probe': probe,
                'brain_region': None if region_label == 'unknown_region' else region_label,
                'region_label': region_label,
                'region_safe': region_label.replace('/', '_').replace('\\', '_').replace(' ', '_'),
                'df_units': region_df,
            })
    return out


def batch_run_by_probe_and_region(
    df_units,
    plot_func,
    *,
    save_dir=None,
    probes=None,
    label=False,
    KSlabel=False,
    bombcell_label=False,
    KSlabel_good=False,
    kslabel_mua=False,
    units_per_group=None,
    include_unknown=True,
    cleanup_between_groups=None,
    skip_existing_save_dir=True,
    **plot_kwargs,
):
    groups = split_units_by_probe_and_region(
        df_units,
        probes=probes,
        label=label,
        KSlabel=KSlabel,
        bombcell_label=bombcell_label,
        KSlabel_good=KSlabel_good,
        kslabel_mua=kslabel_mua,
        include_unknown=include_unknown,
    )
    if cleanup_between_groups is None:
        cleanup_between_groups = save_dir is not None

    results = {}
    for group in groups:
        group_df = group['df_units']
        if units_per_group is not None:
            group_df = group_df.head(int(units_per_group)).copy()
        if group_df.empty:
            continue

        group_key = f"probe{group['probe']}_{group['region_safe']}"
        group_save_dir = Path(save_dir) / group_key if save_dir is not None else None
        if skip_existing_save_dir and group_save_dir is not None and group_save_dir.exists():
            print(f"Skipping {group_key}: save dir already exists at {group_save_dir}")
            if cleanup_between_groups:
                plt.close('all')
                gc.collect()
            continue
        try:
            results[group_key] = plot_func(
                df_units=group_df,
                probeLetter=group['probe'],
                brain_region=group['brain_region'],
                save_dir=group_save_dir,
                **plot_kwargs,
            )
        finally:
            if cleanup_between_groups:
                plt.close('all')
                gc.collect()
    return results


def _iter_unit_ids(df_units, selected_units=None):
    if selected_units is not None:
        return [int(unit) for unit in selected_units]
    cluster_ids = _cluster_id_series(df_units)
    if cluster_ids.notna().any():
        return sorted(cluster_ids.dropna().astype(int).unique().tolist())
    return list(df_units.index)


def _find_unit_row(df_units, unit_id):
    cluster_ids = _cluster_id_series(df_units)
    mask = cluster_ids.eq(float(unit_id))
    if mask.any():
        return df_units.loc[mask].iloc[0]
    if unit_id in df_units.index:
        return df_units.loc[[unit_id]].iloc[0]
    raise ValueError(f'Unit {unit_id} was not found.')


def _resolve_event_times(event_times=None, df_stim=None, epoch1='pellet_delivery_timestamp'):
    if event_times is not None:
        arr = np.asarray(event_times, dtype=float).ravel()
        return arr[np.isfinite(arr)]
    if df_stim is None:
        raise ValueError('Provide event_times or df_stim.')
    if 'stimulus' not in df_stim.columns or 'start_time' not in df_stim.columns:
        raise ValueError('df_stim must contain stimulus and start_time columns.')
    mask = df_stim['stimulus'].astype(str) == str(epoch1)
    if 'optogenetics_LED_state' in df_stim.columns:
        led = pd.to_numeric(df_stim['optogenetics_LED_state'], errors='coerce').fillna(0)
        mask &= led.eq(0)
    arr = pd.to_numeric(df_stim.loc[mask, 'start_time'], errors='coerce').to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _ensure_save_dir(save_dir):
    if save_dir is None:
        return None
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    return save_path


def create_save_dir(save_dir, pre, post):
    pre, post = _coerce_window(pre, post)
    folder = _ensure_save_dir(Path(save_dir) / f'{pre:g}_{post:g}')
    return str(folder)


def _save_figure(fig, save_dir, filename):
    save_path = _ensure_save_dir(save_dir)
    if save_path is None:
        return None
    full_path = save_path / filename
    use_tight_bbox = fig.get_figwidth() < 18 and len(fig.axes) <= 4
    fig.savefig(full_path, bbox_inches='tight' if use_tight_bbox else None)
    return full_path


def _release_figure(fig):
    if fig is None:
        return
    try:
        fig.clf()
    finally:
        plt.close(fig)
        gc.collect()


def _unit_region(row):
    for key in ['Brain_Region', 'brain_region']:
        if key in row.index:
            value = _normalize_region_value(row.get(key))
            if value != 'unknown_region':
                return value
    return 'unknown_region'


def _valid_indices(indices, n_events):
    idx = np.asarray(indices, dtype=int).ravel()
    return idx[(idx >= 0) & (idx < n_events)]


def _baseline_rate(bytrial, pre, bin_size):
    if bytrial.size == 0:
        return 0.0
    baseline_bins = max(1, int(round(pre / bin_size)))
    baseline_bins = min(baseline_bins, bytrial.shape[1])
    return float(np.nanmean(bytrial[:, :baseline_bins])) if baseline_bins > 0 else 0.0


def trial_by_trial(spike_times, event_times, pre, post, bin_size, brain_region=None):
    pre, post = _coerce_window(pre, post)
    if bin_size <= 0:
        raise ValueError('bin_size must be positive.')

    spike_times = np.asarray(spike_times, dtype=float).ravel()
    spike_times = np.sort(spike_times[np.isfinite(spike_times)])
    event_times = np.asarray(event_times, dtype=float).ravel()
    event_times = event_times[np.isfinite(event_times)]

    n_bins = max(1, int(round((pre + post) / float(bin_size))))
    hist_edges = np.linspace(-pre, post, n_bins + 1)
    centers = hist_edges[:-1] + (bin_size / 2.0)

    if event_times.size == 0:
        bytrial = np.zeros((0, n_bins), dtype=np.float32)
        psth = np.zeros(n_bins, dtype=np.float32)
        var = np.zeros(n_bins, dtype=np.float32)
        return psth, var, centers.astype(np.float32, copy=False), bytrial

    bytrial = np.zeros((len(event_times), n_bins), dtype=np.float32)
    for idx, event_time in enumerate(event_times):
        rel_times = spike_times[(spike_times >= event_time - pre) & (spike_times < event_time + post)] - event_time
        hist, _ = np.histogram(rel_times, bins=hist_edges)
        bytrial[idx] = hist.astype(np.float32, copy=False)

    psth = (np.nanmean(bytrial, axis=0) / bin_size).astype(np.float32, copy=False)
    var = (np.nanstd(bytrial, axis=0) / bin_size / np.sqrt(max(len(event_times), 1))).astype(np.float32, copy=False)
    return psth, var, centers.astype(np.float32, copy=False), bytrial


def smooth_data(data, sigma=1.5):
    values = np.asarray(data, dtype=float)
    if values.size == 0:
        return values
    if gaussian_filter1d is not None:
        return gaussian_filter1d(values, sigma=float(sigma))
    window = max(1, int(round(float(sigma) * 3)))
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode='same')


def sort_units_by_firing_rate_change(heatmap_data, time_axis, pre, post, bin_size, sort_by_time=False, smoothing_window=1.5):
    heatmap = np.asarray(heatmap_data, dtype=float)
    if heatmap.ndim != 2 or heatmap.shape[0] == 0:
        return np.array([], dtype=int)
    smoothed = np.array([smooth_data(row, sigma=smoothing_window) for row in heatmap])
    if sort_by_time:
        return np.argsort(np.argmax(smoothed, axis=1))
    magnitudes = np.max(np.abs(np.diff(smoothed, axis=1)), axis=1)
    return np.argsort(-magnitudes)


def make_color_vibrant(color, saturation_increase=0.5, lightness_decrease=0.1):
    rgb = np.array(mcolors.to_rgb(color))
    h, l, s = rgb_to_hls(*rgb)
    s = min(1, s + saturation_increase)
    l = max(0, l - lightness_decrease)
    return mcolors.to_hex(hls_to_rgb(h, l, s))


def _gradient_colors(base_color, n_colors, light_mix=0.7):
    if n_colors <= 0:
        return []
    base = np.array(mcolors.to_rgb(base_color), dtype=float)
    white = np.ones(3, dtype=float)
    if n_colors == 1:
        return [mcolors.to_hex(base)]
    mix_values = np.linspace(float(light_mix), 0.0, int(n_colors))
    colors = []
    for mix in mix_values:
        rgb = base * (1.0 - mix) + white * mix
        colors.append(mcolors.to_hex(np.clip(rgb, 0.0, 1.0)))
    return colors


def _legend_outside(ax, handles, *, loc='center left', bbox_to_anchor=(1.02, 0.5)):
    if not handles:
        return None
    return ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=False,
        borderaxespad=0.0,
    )


def _tight_layout_figure(fig, *, top=0.95, reserve_right=False):
    right = 0.8 if reserve_right else 1.0
    if fig.get_figwidth() >= 18 or len(fig.axes) > 4:
        fig.subplots_adjust(left=0.06, bottom=0.06, right=right, top=top, wspace=0.28, hspace=0.32)
        return
    try:
        fig.tight_layout(rect=[0, 0, right, top])
    except MemoryError:
        fig.subplots_adjust(left=0.06, bottom=0.06, right=right, top=top, wspace=0.28, hspace=0.32)


def _labeled_line_patches(ax):
    handles = []
    for line in ax.lines:
        label = str(line.get_label())
        if not label or label.startswith('_'):
            continue
        handles.append(mpatches.Patch(color=line.get_color(), label=label))
    return handles


def _plot_raster_points(ax, spike_times, event_times, pre, post, color='black', dot_size=4, y_values=None):
    max_y = -1
    for idx, event_time in enumerate(np.asarray(event_times, dtype=float)):
        rel_spikes = spike_times[(spike_times > event_time - pre) & (spike_times < event_time + post)] - event_time
        y_val = y_values[idx] if y_values is not None else idx
        ax.scatter(rel_spikes, np.full(rel_spikes.shape, y_val), marker='|', color=color, s=dot_size, alpha=0.6)
        max_y = max(max_y, y_val)
    return max_y


def _grouped_overlay_legend_handles(trail_indices, color_mapping=None):
    colors = DEFAULT_SPLIT_COLORS.copy()
    if color_mapping:
        colors.update(color_mapping)

    handles = []
    for trial_type, indices in trail_indices.items():
        valid = np.asarray(indices, dtype=int).ravel()
        if valid.size == 0:
            continue
        handles.append(
            mpatches.Patch(
                color=colors.get(trial_type, 'gray'),
                label=str(trial_type).replace('_', ' '),
            )
        )
    return handles


def _plot_grouped_overlay(ax_psth, ax_raster, spike_times, event_times, trail_indices, pre, post, bin_size, color_mapping=None,var_bars=True, dot_size=4, show_legend=True):
    colors = DEFAULT_SPLIT_COLORS.copy()
    if color_mapping:
        colors.update(color_mapping)
    legend_items = []
    max_y = -1
    for trial_type, indices in trail_indices.items():
        valid = _valid_indices(indices, len(event_times))
        if len(valid) == 0:
            continue
        plot_times = np.asarray(event_times, dtype=float)[valid]
        psth, var, edges, bytrial = trial_by_trial(spike_times, plot_times, pre, post, bin_size)
        mean_baseline = _baseline_rate(bytrial, pre, bin_size)
        color = colors.get(trial_type, 'gray')
        _plot_psth(ax_psth, edges, psth, var, mean_baseline, color=color, smooth=True, var_bars=var_bars,smooth_window=5)
        legend_items.append(mpatches.Patch(color=color, label=f'{trial_type} (n={len(valid)})'))
        max_y = max(max_y, _plot_raster_points(ax_raster, spike_times, plot_times, pre, post, color=color, dot_size=dot_size, y_values=valid))
    if show_legend and legend_items:
        _legend_outside(ax_psth, legend_items)
    return max_y


def _segment_event_indices(trail_indices, gap_threshold=5):
    segments = []
    for trial_type, indices in trail_indices.items():
        valid = np.asarray(indices, dtype=int)
        valid = np.sort(valid.astype(int))
        if valid.size == 0:
            continue
        split_points = np.where(np.diff(valid) > int(gap_threshold))[0] + 1
        for segment in np.split(valid, split_points):
            if segment.size == 0:
                continue
            segments.append((trial_type, segment))
    return segments


def _multi_event_columns(fig_cols, *, width_per_col=7.25, fig_height=10.5):
    if fig_cols <= 0:
        raise ValueError('At least one event column is required.')
    return plt.subplots(2, fig_cols, figsize=(width_per_col * fig_cols, fig_height), squeeze=False, sharex='col')


def _heatmap_rows(df_probe, unit_ids, event_times, pre, post, bin_size, normalize_fr=False, max_fr=60, smoothing_sigma=None):
    rows = []
    labels = []
    for unit_id in unit_ids:
        row = _find_unit_row(df_probe, unit_id)
        spike_times = np.asarray(row['spike_times'], dtype=float)
        _, _, centers, bytrial = trial_by_trial(spike_times, event_times, pre, post, bin_size)
        heat = np.sum(bytrial, axis=0, dtype=np.float32)
        if smoothing_sigma is not None:
            heat = smooth_data(heat, sigma=smoothing_sigma).astype(np.float32, copy=False)
        if normalize_fr:
            mean_fr = np.mean(heat)
            std_fr = np.std(heat)
            heat = (heat - mean_fr) / std_fr if std_fr != 0 else np.zeros_like(heat, dtype=np.float32)
        else:
            heat = np.clip(heat, 0, max_fr).astype(np.float32, copy=False)
        rows.append(np.asarray(heat, dtype=np.float32))
        labels.append(int(unit_id))
    if not rows:
        return np.empty((0, 0), dtype=np.float32), np.array([], dtype=int), np.array([], dtype=np.float32)
    return np.asarray(rows, dtype=np.float32), np.asarray(labels), np.asarray(centers, dtype=np.float32)

def _safe_value_text(value, default='NA'):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        if value.size == 1:
            value = value.reshape(-1)[0]
        else:
            return f'array{tuple(value.shape)}'

    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        pass

    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return f'{float(value):.2f}'
    return str(value)


def _format_float(value, digits=2):
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return str(value)

    if arr.size == 0:
        return 'nan'

    scalar = float(np.nanmean(arr))
    if not np.isfinite(scalar):
        return 'nan'
    return f'{scalar:.{digits}f}'


def _join_title_parts(parts):
    return ' | '.join(str(part) for part in parts if part not in [None, ''])


def _ks_label_to_str(label_value):
    if label_value == 2:
        return 'good'
    if label_value == 1:
        return 'mua'
    if label_value == 0:
        return 'noise'
    return _safe_value_text(label_value)


def _manual_label_to_str(label_value):
    if label_value == 2:
        return 'good'
    if label_value == 1:
        return 'mua'
    if label_value == 0:
        return 'No Manual Label Added'
    return _safe_value_text(label_value)


def _get_event_label(namespace, event_value, fallback_label):
    if isinstance(event_value, str):
        return event_value

    for name, value in namespace.items():
        if value is event_value:
            return name

    event_array = np.asarray(event_value)
    for name, value in namespace.items():
        try:
            value_array = np.asarray(value)
        except Exception:
            continue

        if value_array.shape == event_array.shape and np.array_equal(value_array, event_array):
            return name

    return fallback_label


def _trial_type_style(trial_type):
    trial_type = trial_type or 'all_trials'
    style_map = {
        'all_trials': {
            'psth_color': 'black',
            'raster_color': 'black',
            'display_name': 'all_trials',
        },
        'stimulation_trials': {
            'psth_color': 'blue',
            'raster_color': 'blue',
            'display_name': 'stimulation_trials',
        },
        'washout_trials': {
            'psth_color': 'red',
            'raster_color': 'red',
            'display_name': 'washout_trials',
        },
    }
    if trial_type not in style_map:
        raise ValueError(f'Unsupported trial_type: {trial_type}')

    return {'trial_type': trial_type, **style_map[trial_type]}


def _resolve_event_source(event_source, namespace, fallback_label):
    if isinstance(event_source, str):
        if event_source not in namespace:
            raise KeyError(f'Event source {event_source!r} was not found in namespace. Use a string variable name or pass a dict value with event_times.')
        event_times = np.asarray(namespace[event_source])
        event_label = event_source
    else:
        event_times = np.asarray(event_source)
        event_label = _get_event_label(namespace, event_source, fallback_label)

    return event_times, event_label
def build_event_name_subplots(*entries, namespace=None):
    if namespace is None:
        namespace = globals()

    event_specs = {}
    for idx, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValueError('Each entry must be a 2-item tuple: (event_times, trial_type).')

        event_source, trial_type = entry
        event_label = _get_event_label(namespace, event_source, f'event_times_{idx + 1}')
        event_specs[event_label] = {
            'event_times': np.asarray(event_source),
            'event_name': event_label,
            'trial_type': trial_type,
        }

    return event_specs


def _normalize_event_specs(event_name_subplots=None, namespace=None, event_times_subplots=None, event_names=None, default_trial_type='all_trials'):
    if namespace is None:
        namespace = globals()

    event_entries = []

    if event_name_subplots is not None:
        if isinstance(event_name_subplots, dict):
            iterable = list(event_name_subplots.items())
        elif isinstance(event_name_subplots, (list, tuple)):
            iterable = list(event_name_subplots)
        else:
            raise ValueError('event_name_subplots must be a dict or a list of (event_source, trial_type) pairs.')

        for idx, pair in enumerate(iterable):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError('Each event specification must be a 2-item pair: (event_source, trial_type_or_spec).')

            raw_key, raw_value = pair
            fallback_label = f'event_times_{idx + 1}'
            if isinstance(raw_value, dict):
                trial_type = raw_value.get('trial_type', raw_value.get('trial_group', default_trial_type))
                event_source = raw_value.get('event_times', raw_key)
                event_label = raw_value.get('event_name', raw_value.get('label'))
            else:
                trial_type = raw_value
                event_source = raw_key
                event_label = raw_key if isinstance(raw_key, str) else None

            event_times, resolved_label = _resolve_event_source(event_source, namespace, fallback_label)
            if event_label is None:
                event_label = resolved_label

            event_entries.append({
                'event_label': event_label,
                'event_times': event_times,
                **_trial_type_style(trial_type),
            })

        return event_entries

    if event_times_subplots is None:
        raise ValueError('Provide event_name_subplots or event_times_subplots.')

    if event_names is not None and len(event_names) != len(event_times_subplots):
        raise ValueError('event_names must match the length of event_times_subplots.')

    for idx, event_source in enumerate(event_times_subplots):
        label_source = event_names[idx] if event_names is not None else event_source
        event_times, _ = _resolve_event_source(event_source, namespace, f'event_times_{idx + 1}')
        if event_names is not None:
            _, event_label = _resolve_event_source(label_source, namespace, f'event_times_{idx + 1}')
        else:
            event_label = _get_event_label(namespace, event_source, f'event_times_{idx + 1}')

        event_entries.append({
            'event_label': event_label,
            'event_times': event_times,
            **_trial_type_style(default_trial_type),
        })

    return event_entries

def _collect_unit_plot_metadata(df_units, probeLetter, cluster_id):
    df1 = df_units[(df_units.probe == probeLetter) & (df_units.cluster_id == cluster_id)]
    if df1.empty:
        raise ValueError(f'No unit found for probe {probeLetter} and cluster_id {cluster_id}.')

    row = df1.iloc[0]
    brain_region = row.get('Brain_Region', 'unknown_region')
    if pd.isna(brain_region) or str(brain_region) in ['None', 'nan']:
        brain_region = 'unknown_region'

    spike_times = np.asarray(row['spike_times'])
    extra_fields = [
        ('unit', 'unit'),
        ('unit_id', 'unit_id'),
        ('depth', 'depth'),
        ('depth_um', 'depth_um'),
        ('ch', 'ch'),
        ('peak_channel', 'peak_ch'),
    ]
    extra_parts = []
    for column_name, label in extra_fields:
        if column_name not in df1.columns:
            continue

        value_text = _safe_value_text(row.get(column_name), default=None)
        if value_text is not None:
            extra_parts.append(f'{label}: {value_text}')

    metadata = {
        'df1': df1,
        'probeLetter': probeLetter,
        'cluster_id': cluster_id,
        'brain_region': str(brain_region),
        'brain_region_safe': str(brain_region).replace('/', '_').replace('\\', '_').replace(' ', '_'),
        'spike_times': spike_times,
        'spike_times_count': int(spike_times.size),
        'bc_unitType': _safe_value_text(row.get('bc_unitType')),
        'KS_label': row.get('KSlabel', np.nan),
        'manual_label': row.get('label', np.nan),
        'extra_parts': extra_parts,
    }
    metadata['KS_label_str'] = _ks_label_to_str(metadata['KS_label'])
    metadata['manual_label_str'] = _manual_label_to_str(metadata['manual_label'])
    return metadata


def _summarize_trial_response(psth, edges, bytrial, pre, bin_size):
    bytrial = np.asarray(bytrial)
    psth = np.asarray(psth)
    edges = np.asarray(edges)

    baseline_bins = max(1, int(pre / bin_size))
    baseline_bins = min(baseline_bins, bytrial.shape[1]) if bytrial.ndim == 2 and bytrial.shape[1] else 1

    baseline_window = bytrial[:, :baseline_bins]
    if bytrial.shape[1] > baseline_bins:
        response_window = bytrial[:, baseline_bins:]
    else:
        response_window = bytrial[:, -1:]

    mean_baseline = float(np.nanmean(baseline_window))
    std_baseline = float(np.nanstd(baseline_window))
    mean_response = float(np.nanmean(response_window))
    std_response = float(np.nanstd(response_window))
    delta_mean = mean_response - mean_baseline

    trial_response_mean = np.nanmean(response_window, axis=1)
    if np.isfinite(std_baseline) and std_baseline > 0:
        trials_z = (trial_response_mean - mean_baseline) / std_baseline
    else:
        trials_z = np.full(trial_response_mean.shape, np.nan)

    psth_delta = psth - mean_baseline
    finite_delta = np.isfinite(psth_delta)
    finite_psth = np.isfinite(psth)

    if finite_delta.any():
        peak_idx = int(np.nanargmax(psth_delta))
        trough_idx = int(np.nanargmin(psth_delta))
        peak_delta_hz = float(psth_delta[peak_idx])
        trough_delta_hz = float(psth_delta[trough_idx])
        peak_time_s = float(edges[peak_idx]) if peak_idx < edges.size else np.nan
        trough_time_s = float(edges[trough_idx]) if trough_idx < edges.size else np.nan
    else:
        peak_delta_hz = np.nan
        trough_delta_hz = np.nan
        peak_time_s = np.nan
        trough_time_s = np.nan

    peak_rate_hz = float(np.nanmax(psth)) if finite_psth.any() else np.nan

    return {
        'trial_count': int(bytrial.shape[0]),
        'bin_count': int(bytrial.shape[1]),
        'baseline_bins': int(baseline_bins),
        'response_bins': int(response_window.shape[1]),
        'mean_baseline': mean_baseline,
        'std_baseline': std_baseline,
        'mean_response': mean_response,
        'std_response': std_response,
        'delta_mean': delta_mean,
        'trial_z_mean': float(np.nanmean(trials_z)) if trials_z.size else np.nan,
        'trial_z_min': float(np.nanmin(trials_z)) if trials_z.size else np.nan,
        'trial_z_max': float(np.nanmax(trials_z)) if trials_z.size else np.nan,
        'peak_delta_hz': peak_delta_hz,
        'peak_time_s': peak_time_s,
        'trough_delta_hz': trough_delta_hz,
        'trough_time_s': trough_time_s,
        'peak_rate_hz': peak_rate_hz,
    }


def _aggregate_summary(summaries):
    if not summaries:
        return None

    aggregate = {
        'total_events': int(sum(summary.get('n_events', 0) for summary in summaries)),
        'total_trials': int(sum(summary.get('trial_count', 0) for summary in summaries)),
    }
    summary_keys = [
        'mean_baseline',
        'std_baseline',
        'mean_response',
        'std_response',
        'delta_mean',
        'trial_z_mean',
        'peak_delta_hz',
        'peak_rate_hz',
    ]
    for key in summary_keys:
        values = np.asarray([summary.get(key, np.nan) for summary in summaries], dtype=float)
        finite_values = values[np.isfinite(values)]
        aggregate[f'{key}_mean'] = float(np.nanmean(values)) if values.size else np.nan
        aggregate[f'{key}_max'] = float(np.nanmax(finite_values)) if finite_values.size else np.nan

    return aggregate


def _build_unit_header_lines(metadata, pre, post, bin_size, extra_parts=None):
    header_line = _join_title_parts([
        f"cluster_id: {metadata['cluster_id']}",
        f"probe: {metadata['probeLetter']}",
        f"BR: {metadata['brain_region']}",
        f"bc_unitType: {metadata['bc_unitType']}",
        f"KS_label: {metadata['KS_label_str']}",
        f"Manual_label: {metadata['manual_label_str']}",
        f"n_spikes: {metadata['spike_times_count']}",
    ])
    settings_parts = list(metadata['extra_parts'])
    settings_parts.extend([
        f'pre: {_format_float(pre)} s',
        f'post: {_format_float(post)} s',
        f'bin_size: {_format_float(bin_size, digits=3)} s',
    ])
    if extra_parts:
        settings_parts.extend(extra_parts)
    return [header_line, _join_title_parts(settings_parts)]


def _smooth_psth_trace(psth_values, window_size=5):
    if window_size <= 1 or len(psth_values) < window_size:
        return psth_values

    kernel = np.ones(window_size, dtype=float) / window_size
    return np.convolve(psth_values, kernel, mode='same')


def _plot_psth(ax, edges, psth, var, mean_baseline, color='b', smooth=True,var_bars=True, smooth_window=5, fill_alpha=0.2):
    center = psth - mean_baseline
    upper = psth + var - mean_baseline
    lower = psth - var - mean_baseline

    if smooth:
        center = _smooth_psth_trace(center, window_size=smooth_window)
        upper = _smooth_psth_trace(upper, window_size=smooth_window)
        lower = _smooth_psth_trace(lower, window_size=smooth_window)

    ax.plot(edges, center, color=color)

    if var_bars:
        ax.fill_between(edges, upper, lower, alpha=fill_alpha, color=color)


def singleUnit_psth_raster_test(df_units, probeLetter,cluster_id, event_name_subplots=None, pre=None, post=None, bin_size=0.01, namespace=None, smooth=True,var_bars=True, smooth_window=5, event_times=None, event_label=None, trial_type='all_trials'):
    if namespace is None:
        namespace = globals()

    if event_name_subplots is None:
        if event_times is None:
            raise ValueError('Provide event_name_subplots or event_times.')
        event_label_resolved = event_label if event_label is not None else _get_event_label(namespace, event_times, 'event_times')
        event_name_subplots = [
            (
                event_label_resolved,
                {
                    'event_times': np.asarray(event_times),
                    'trial_type': trial_type,
                    'event_name': event_label_resolved,
                },
            )
        ]

    event_entries = _normalize_event_specs(event_name_subplots=event_name_subplots, namespace=namespace)
    if len(event_entries) != 1:
        raise ValueError('singleUnit_psth_raster_test expects exactly one event specification.')

    event_entry = event_entries[0]
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(10, 8)

    metadata = _collect_unit_plot_metadata(df_units, probeLetter, cluster_id)

    spike_times = metadata['spike_times']
    event_times = event_entry['event_times']
    psth, var, edges, bytrial = trial_by_trial(spike_times, event_times, pre, post, bin_size)
    stats = _summarize_trial_response(psth, edges, bytrial, pre, bin_size)
    stats['n_events'] = int(len(event_times))

    ax[0].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax[1].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    _plot_psth(ax[0], edges, psth, var, stats['mean_baseline'], color=event_entry['psth_color'], smooth=smooth,var_bars=var_bars, smooth_window=smooth_window)
    ax[0].set_ylabel('Firing Rate [Hz]')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_xlim(-pre, post)
    ax[0].set_title('\n'.join(_build_unit_header_lines(
        metadata,
        pre,
        post,
        bin_size,
        extra_parts=[
            f"event: {event_entry['event_label']}",
            f"trial_type: {event_entry['display_name']}",
            f"n_events: {stats['n_events']}",
            f"n_trials: {stats['trial_count']}",
            f"smooth: {'yes' if smooth else 'no'}",
            f"smooth_window: {smooth_window}",
        ]
    ) + [
        _join_title_parts([
            f"mean_baseline: {_format_float(stats['mean_baseline'])}",
            f"std_baseline: {_format_float(stats['std_baseline'])}",
            f"mean_response: {_format_float(stats['mean_response'])}",
            f"std_response: {_format_float(stats['std_response'])}",
        ]),
        _join_title_parts([
            f"delta_mean: {_format_float(stats['delta_mean'])}",
            f"trials_zMean: {_format_float(stats['trial_z_mean'])}",
            f"peak_delta: {_format_float(stats['peak_delta_hz'])} Hz @ {_format_float(stats['peak_time_s'])} s",
            f"peak_rate: {_format_float(stats['peak_rate_hz'])} Hz",
        ]),
    ]))

    for t, time in enumerate(event_times):
        trial_spikes = spike_times[(spike_times > time - pre) & (spike_times < time + post)]
        trial_spikes = trial_spikes - time
        ax[1].scatter(trial_spikes, [t] * len(trial_spikes), marker='|', color=event_entry['raster_color'], s=1, alpha=0.6)

    ax[1].set_ylabel('Trial')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_xlim(-pre, post)
    ax[1].set_title(_join_title_parts([
        event_entry['event_label'],
        'Raster Plot',
        f"trial_type: {event_entry['display_name']}",
        f"n_trials: {stats['trial_count']}",
        f"n_events: {stats['n_events']}",
        f"trial_z range: {_format_float(stats['trial_z_min'])} to {_format_float(stats['trial_z_max'])}",
    ]))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax


def singleUnit_psth_raster_subplots(df_units, probeLetter, cluster_id, event_name_subplots=None, pre=None, post=None, bin_size=0.01, namespace=None, smooth=True, var_bars=True, smooth_window=5, event_times_subplots=None, event_names=None):
    if namespace is None:
        namespace = globals()

    event_entries = _normalize_event_specs(
        event_name_subplots=event_name_subplots,
        namespace=namespace,
        event_times_subplots=event_times_subplots,
        event_names=event_names,
    )

    metadata = _collect_unit_plot_metadata(df_units, probeLetter, cluster_id)
    spike_times = metadata['spike_times']

    num_columns = len(event_entries)
    fig, ax = plt.subplots(2, num_columns, figsize=(6 * num_columns, 8), squeeze=False)

    panel_summaries = []
    trial_type_summary = []
    for col, event_entry in enumerate(event_entries):
        event_times = event_entry['event_times']
        subplot_title = event_entry['event_label']

        ax[0, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax[1, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax[0, col].set_xlim(-pre, post)
        ax[1, col].set_xlim(-pre, post)

        if len(event_times) == 0:
            no_event_title = '\n'.join([
                subplot_title,
                event_entry['display_name'],
                _join_title_parts(['n_events: 0', 'n_trials: 0', 'no events']),
            ])
            ax[0, col].set_title(no_event_title)
            ax[0, col].set_ylabel('Firing Rate [Hz]')
            ax[0, col].set_xlabel('Time [s]')
            ax[1, col].set_ylabel('Trial')
            ax[1, col].set_xlabel('Time [s]')
            ax[1, col].set_title(no_event_title)
            continue

        psth, var, edges, bytrial = trial_by_trial(spike_times, event_times, pre, post, bin_size)
        stats = _summarize_trial_response(psth, edges, bytrial, pre, bin_size)
        stats['n_events'] = int(len(event_times))
        panel_summaries.append(stats)
        trial_type_summary.append(f"{subplot_title}: {event_entry['display_name']}")

        _plot_psth(ax[0, col], edges, psth, var, stats['mean_baseline'], color=event_entry['psth_color'], smooth=smooth, var_bars=var_bars,smooth_window=smooth_window)
        ax[0, col].set_ylabel('Firing Rate [Hz]')
        ax[0, col].set_xlabel('Time [s]')
        ax[0, col].set_title('\n'.join([
            subplot_title,
            event_entry['display_name'],
            _join_title_parts([
                f"n_events: {stats['n_events']}",
                f"n_trials: {stats['trial_count']}",
                f"mean_baseline: {_format_float(stats['mean_baseline'])}",
                f"std_baseline: {_format_float(stats['std_baseline'])}",
            ]),
            _join_title_parts([
                f"mean_response: {_format_float(stats['mean_response'])}",
                f"delta_mean: {_format_float(stats['delta_mean'])}",
                f"trials_zMean: {_format_float(stats['trial_z_mean'])}",
                f"peak_delta: {_format_float(stats['peak_delta_hz'])} Hz @ {_format_float(stats['peak_time_s'])} s",
            ]),
        ]))

        for t, time in enumerate(event_times):
            trial_spikes = spike_times[(spike_times > time - pre) & (spike_times < time + post)]
            trial_spikes = trial_spikes - time
            ax[1, col].scatter(trial_spikes, [t] * len(trial_spikes), marker='|', color=event_entry['raster_color'], s=1, alpha=0.6)

        ax[1, col].set_ylabel('Trial')
        ax[1, col].set_xlabel('Time [s]')
        ax[1, col].set_title('\n'.join([
            subplot_title,
            event_entry['display_name'],
            _join_title_parts([
                'Raster Plot',
                f"n_trials: {stats['trial_count']}",
                f"trial_z range: {_format_float(stats['trial_z_min'])} to {_format_float(stats['trial_z_max'])}",
            ]),
        ]))

    aggregate = _aggregate_summary(panel_summaries)
    suptitle_lines = _build_unit_header_lines(
        metadata,
        pre,
        post,
        bin_size,
        extra_parts=[
            f'event_sets: {num_columns}',
            f"smooth: {'yes' if smooth else 'no'}",
            f'smooth_window: {smooth_window}',
            f"total_events: {aggregate['total_events'] if aggregate else 0}",
            f"total_trials: {aggregate['total_trials'] if aggregate else 0}",
        ],
    )
    if trial_type_summary:
        suptitle_lines.append(_join_title_parts(trial_type_summary))
    if aggregate:
        suptitle_lines.extend([
            _join_title_parts([
                f"mean_baseline(avg): {_format_float(aggregate['mean_baseline_mean'])}",
                f"std_baseline(avg): {_format_float(aggregate['std_baseline_mean'])}",
                f"mean_response(avg): {_format_float(aggregate['mean_response_mean'])}",
                f"delta_mean(avg): {_format_float(aggregate['delta_mean_mean'])}",
            ]),
            _join_title_parts([
                f"trials_zMean(avg): {_format_float(aggregate['trial_z_mean_mean'])}",
                f"peak_delta(max): {_format_float(aggregate['peak_delta_hz_max'])} Hz",
                f"peak_rate(max): {_format_float(aggregate['peak_rate_hz_max'])} Hz",
            ]),
        ])
    fig.suptitle('\n'.join(suptitle_lines))

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    return fig, ax


def singleUnit_psth_raster_subplots_stim_seperated(df_units, probeLetter, cluster_id, event_name_subplots=None, trail_indices=None, pre=None, post=None, bin_size=0.01, event_names=None, namespace=None, smooth=True, var_bars=True, smooth_window=5, show_data_in_title=True, save_figure=False, output_root=None, event_times_subplots=None, save_dir=None):
    if trail_indices is None:
        raise ValueError('trail_indices is required for baseline/stimulation/washout separation.')

    if namespace is None:
        namespace = globals()

    def _valid_indices(indices, n_events):
        indices = np.asarray(indices, dtype=int)
        return indices[(indices >= 0) & (indices < n_events)]

    def _set_subplot_title(axis, event_label, row_title, detail_lines=None, always_show_details=False):
        title_lines = [event_label, row_title]
        if detail_lines and (show_data_in_title or always_show_details):
            title_lines.extend(detail_lines)
        axis.set_title('\n'.join(title_lines))

    event_entries = _normalize_event_specs(
        event_name_subplots=event_name_subplots,
        namespace=namespace,
        event_times_subplots=event_times_subplots,
        event_names=event_names,
    )

    metadata = _collect_unit_plot_metadata(df_units, probeLetter, cluster_id)
    spike_times = metadata['spike_times']

    split_color_mapping = {
        'baseline': 'green',
        'optical_stim': 'blue',
        'no_optical_stim': 'red',
    }
    row_specs = [
        ('baseline', 'Baseline PSTH'),
        ('optical_stim', 'Stimulation PSTH'),
        ('no_optical_stim', 'Washout PSTH'),
    ]
    row_key_by_trial_type = {
        'stimulation_trials': 'optical_stim',
        'washout_trials': 'no_optical_stim',
    }

    num_columns = len(event_entries)
    fig, ax = plt.subplots(5, num_columns, figsize=(6 * num_columns, 18), squeeze=False, sharex='col')

    all_trial_summaries = []
    phase_trial_counts = {key: 0 for key, _ in row_specs}
    total_events = 0
    trial_type_summary = []

    for col, event_entry in enumerate(event_entries):
        subplot_title = event_entry['event_label']
        event_times = event_entry['event_times']
        trial_type = event_entry['trial_type']
        total_events += int(len(event_times))
        trial_type_summary.append(f"{subplot_title}: {event_entry['display_name']}")

        for row in range(5):
            ax[row, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax[row, col].set_xlim(-pre, post)

        if len(event_times) == 0:
            for row, row_title in enumerate(['Baseline PSTH', 'Stimulation PSTH', 'Washout PSTH', 'All Trials PSTH', 'Raster Plot']):
                _set_subplot_title(ax[row, col], subplot_title, row_title, ['no events'], always_show_details=(row == 4))
            continue

        for row, (trial_key, row_title) in enumerate(row_specs):
            if trial_type == 'all_trials':
                valid_indices = _valid_indices(trail_indices.get(trial_key, []), len(event_times))
                plot_times = event_times[valid_indices]
                color = split_color_mapping.get(trial_key, 'gray')
            else:
                is_active_row = row_key_by_trial_type.get(trial_type) == trial_key
                plot_times = event_times if is_active_row else np.asarray([])
                color = event_entry['psth_color']

            phase_trial_counts[trial_key] += int(len(plot_times))

            if len(plot_times) == 0:
                _set_subplot_title(ax[row, col], subplot_title, row_title, ['no trials'])
            else:
                psth, var, edges, bytrial = trial_by_trial(spike_times, plot_times, pre, post, bin_size)
                stats = _summarize_trial_response(psth, edges, bytrial, pre, bin_size)
                stats['n_events'] = int(len(plot_times))
                _plot_psth(ax[row, col], edges, psth, var, stats['mean_baseline'], color=color, smooth=smooth, var_bars=var_bars,smooth_window=smooth_window)
                _set_subplot_title(
                    ax[row, col],
                    subplot_title,
                    row_title,
                    [
                        event_entry['display_name'],
                        _join_title_parts([
                            f"n_trials: {stats['trial_count']}",
                            f"mean_baseline: {_format_float(stats['mean_baseline'])}",
                            f"std_baseline: {_format_float(stats['std_baseline'])}",
                        ]),
                        _join_title_parts([
                            f"mean_response: {_format_float(stats['mean_response'])}",
                            f"delta_mean: {_format_float(stats['delta_mean'])}",
                            f"trials_zMean: {_format_float(stats['trial_z_mean'])}",
                        ]),
                    ],
                )

            ax[row, col].set_ylabel('Firing Rate [Hz]')
            ax[row, col].set_xlabel('Time [s]')

        psth, var, edges, bytrial = trial_by_trial(spike_times, event_times, pre, post, bin_size)
        all_stats = _summarize_trial_response(psth, edges, bytrial, pre, bin_size)
        all_stats['n_events'] = int(len(event_times))
        all_trial_summaries.append(all_stats)
        all_trials_color = 'black' if trial_type == 'all_trials' else event_entry['psth_color']
        _plot_psth(ax[3, col], edges, psth, var, all_stats['mean_baseline'], color=all_trials_color, smooth=smooth, var_bars=var_bars,smooth_window=smooth_window)
        ax[3, col].set_ylabel('Firing Rate [Hz]')
        ax[3, col].set_xlabel('Time [s]')
        _set_subplot_title(
            ax[3, col],
            subplot_title,
            'All Trials PSTH',
            [
                event_entry['display_name'],
                _join_title_parts([
                    f"n_trials: {all_stats['trial_count']}",
                    f"mean_baseline: {_format_float(all_stats['mean_baseline'])}",
                    f"std_baseline: {_format_float(all_stats['std_baseline'])}",
                ]),
                _join_title_parts([
                    f"mean_response: {_format_float(all_stats['mean_response'])}",
                    f"delta_mean: {_format_float(all_stats['delta_mean'])}",
                    f"trials_zMean: {_format_float(all_stats['trial_z_mean'])}",
                    f"peak_delta: {_format_float(all_stats['peak_delta_hz'])} Hz @ {_format_float(all_stats['peak_time_s'])} s",
                ]),
            ],
        )

        max_trial_idx = -1
        trial_count_labels = []
        if trial_type == 'all_trials':
            for trial_key, _ in row_specs:
                valid_indices = _valid_indices(trail_indices.get(trial_key, []), len(event_times))
                color = split_color_mapping.get(trial_key, 'gray')
                trial_count_labels.append(f'{trial_key}: {len(valid_indices)}')

                for t_idx in valid_indices:
                    time = event_times[t_idx]
                    trial_spikes = spike_times[(spike_times > time - pre) & (spike_times < time + post)]
                    trial_spikes = trial_spikes - time
                    ax[4, col].scatter(trial_spikes, [t_idx] * len(trial_spikes), marker='|', color=color, s=4, alpha=0.6)
                    max_trial_idx = max(max_trial_idx, t_idx)
        else:
            trial_count_labels.append(f'{trial_type}: {len(event_times)}')
            for t_idx, time in enumerate(event_times):
                trial_spikes = spike_times[(spike_times > time - pre) & (spike_times < time + post)]
                trial_spikes = trial_spikes - time
                ax[4, col].scatter(trial_spikes, [t_idx] * len(trial_spikes), marker='|', color=event_entry['raster_color'], s=4, alpha=0.6)
                max_trial_idx = max(max_trial_idx, t_idx)

        ax[4, col].set_ylabel('Trial')
        ax[4, col].set_xlabel('Time [s]')
        _set_subplot_title(
            ax[4, col],
            subplot_title,
            'Raster Plot',
            [
                event_entry['display_name'],
                f"all_trials: {len(event_times)}",
                _join_title_parts(trial_count_labels),
            ],
            always_show_details=True,
        )
        if max_trial_idx >= 0:
            ax[4, col].set_ylim(-1, max_trial_idx + 1)

    aggregate = _aggregate_summary(all_trial_summaries)
    suptitle_lines = _build_unit_header_lines(
        metadata,
        pre,
        post,
        bin_size,
        extra_parts=[
            f'event_sets: {num_columns}',
            f"smooth: {'yes' if smooth else 'no'}",
            f'smooth_window: {smooth_window}',
            f'total_events: {total_events}',
            f"show_data_in_title: {'yes' if show_data_in_title else 'no'}",
        ],
    )
    if trial_type_summary:
        suptitle_lines.append(_join_title_parts(trial_type_summary))
    suptitle_lines.append(_join_title_parts([
        f"baseline_trials: {phase_trial_counts['baseline']}",
        f"stim_trials: {phase_trial_counts['optical_stim']}",
        f"washout_trials: {phase_trial_counts['no_optical_stim']}",
    ]))
    if aggregate:
        suptitle_lines.append(_join_title_parts([
            f"all_trials mean_baseline(avg): {_format_float(aggregate['mean_baseline_mean'])}",
            f"mean_response(avg): {_format_float(aggregate['mean_response_mean'])}",
            f"delta_mean(avg): {_format_float(aggregate['delta_mean_mean'])}",
            f"trials_zMean(avg): {_format_float(aggregate['trial_z_mean_mean'])}",
            f"peak_delta(max): {_format_float(aggregate['peak_delta_hz_max'])} Hz",
        ]))
    fig.suptitle('\n'.join(suptitle_lines))
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    # if save_figure:
    #     if output_root is None:
    #         output_root = Path.cwd().resolve() / 'processed_data' / 'psth_raster_output'

    #     fig_filename = Path(output_root) / 'singleUnit_psth_raster_subplots_stim_seperated' / f"probe{probeLetter}_{metadata['brain_region_safe']}" / f"clusterID_{int(cluster_id)}_probe{probeLetter}_{metadata['brain_region_safe']}.png"
    #     fig_filename.parent.mkdir(parents=True, exist_ok=True)
    #     fig.savefig(fig_filename, bbox_inches='tight')

    if save_dir is not None:
        fig_filename = Path(save_dir) / f"singleUnit_psth_raster_subplots_stim_seperated_probe{probeLetter}_clusterID_{int(cluster_id)}.png"
        fig_filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_filename, bbox_inches='tight')

    return fig, ax


def singleUnit_psth_raster_all_events_seperated(df_units, probeLetter, cluster_id, variables, trail_indices, pre, post, bin_size=0.01, namespace=None, smooth=True, var_bars=True, smooth_window=5, show_data_in_title=True, dot_size=0.5, save_dir=None):
    if not variables:
        raise ValueError('variables is required.')
    if trail_indices is None:
        raise ValueError('trail_indices is required.')
    if namespace is None:
        namespace = globals()

    metadata = _collect_unit_plot_metadata(df_units, probeLetter, cluster_id)
    spike_times = metadata['spike_times']

    event_entries = []
    for idx, (event_label, event_source) in enumerate(variables.items()):
        event_times, resolved_label = _resolve_event_source(event_source, namespace, f'event_times_{idx + 1}')
        event_entries.append({
            'event_label': str(event_label) if isinstance(event_label, str) else resolved_label,
            'event_times': np.asarray(event_times, dtype=float),
        })

    fig, ax = _multi_event_columns(len(event_entries), width_per_col=8.5, fig_height=12.0)
    legend_handles = _grouped_overlay_legend_handles(trail_indices)
    all_summaries = []
    total_events = 0
    phase_trial_counts = {key: 0 for key in trail_indices.keys()}

    for col, event_entry in enumerate(event_entries):
        event_label = event_entry['event_label']
        event_times = event_entry['event_times']
        total_events += int(len(event_times))

        ax[0, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax[1, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax[0, col].set_xlim(-abs(pre), abs(post))
        ax[1, col].set_xlim(-abs(pre), abs(post))
        ax[0, col].set_ylabel('Firing Rate [Hz]')
        ax[1, col].set_ylabel('Trial')
        ax[1, col].set_xlabel('Time [s]')

        if len(event_times) == 0:
            ax[0, col].set_title('\n'.join([event_label, 'no events']))
            ax[1, col].set_title('\n'.join([event_label, 'Raster Plot', 'no events']))
            continue

        event_summaries = []
        trial_count_parts = []
        max_y = -1
        for trial_type, indices in trail_indices.items():
            valid = _valid_indices(indices, len(event_times))
            phase_trial_counts[trial_type] = phase_trial_counts.get(trial_type, 0) + int(len(valid))
            trial_count_parts.append(f'{trial_type}: {len(valid)}')
            if len(valid) == 0:
                continue

            plot_times = event_times[valid]
            psth, var, edges, bytrial = trial_by_trial(spike_times, plot_times, abs(pre), abs(post), bin_size)
            stats = _summarize_trial_response(psth, edges, bytrial, abs(pre), bin_size)
            stats['n_events'] = int(len(plot_times))
            stats['trial_type'] = str(trial_type)
            event_summaries.append(stats)
            all_summaries.append(stats)

            color = DEFAULT_SPLIT_COLORS.get(trial_type, 'gray')
            _plot_psth(
                ax[0, col],
                edges,
                psth,
                var,
                stats['mean_baseline'],
                color=color,
                smooth=smooth,
                var_bars=var_bars,
                smooth_window=smooth_window,
            )
            max_y = max(
                max_y,
                _plot_raster_points(
                    ax[1, col],
                    spike_times,
                    plot_times,
                    abs(pre),
                    abs(post),
                    color=color,
                    dot_size=max(0.5, dot_size * 6),
                    y_values=valid,
                ),
            )

        title_lines = [event_label]
        if show_data_in_title:
            title_lines.append(_join_title_parts([
                f'total_events: {len(event_times)}',
                f'trial_groups: {len(event_summaries)}',
            ]))
            if trial_count_parts:
                title_lines.append(_join_title_parts(trial_count_parts))
            aggregate = _aggregate_summary(event_summaries)
            if aggregate:
                title_lines.append(_join_title_parts([
                    f"mean_response(avg): {_format_float(aggregate['mean_response_mean'])}",
                    f"delta_mean(avg): {_format_float(aggregate['delta_mean_mean'])}",
                    f"trials_zMean(avg): {_format_float(aggregate['trial_z_mean_mean'])}",
                    f"peak_delta(max): {_format_float(aggregate['peak_delta_hz_max'])} Hz",
                ]))
        ax[0, col].set_title('\n'.join(title_lines))

        raster_title_lines = [
            event_label,
            'Raster Plot',
            _join_title_parts([f'total_events: {len(event_times)}'] + trial_count_parts),
        ]
        ax[1, col].set_title('\n'.join(raster_title_lines))
        if max_y >= 0:
            ax[1, col].set_ylim(-1, max_y + 1)

    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc='center left',
            bbox_to_anchor=(0.82, 0.5),
            frameon=False,
            title='Trial group',
        )

    aggregate = _aggregate_summary(all_summaries)
    suptitle_lines = _build_unit_header_lines(
        metadata,
        pre,
        post,
        bin_size,
        extra_parts=[
            f'event_sets: {len(event_entries)}',
            f'total_events: {total_events}',
            f"smooth: {'yes' if smooth else 'no'}",
            f'smooth_window: {smooth_window}',
            f'var_bars: {"yes" if var_bars else "no"}',
        ],
    )
    if phase_trial_counts:
        suptitle_lines.append(_join_title_parts([f'{key}: {count}' for key, count in phase_trial_counts.items()]))
    if aggregate:
        suptitle_lines.append(_join_title_parts([
            f"mean_baseline(avg): {_format_float(aggregate['mean_baseline_mean'])}",
            f"mean_response(avg): {_format_float(aggregate['mean_response_mean'])}",
            f"delta_mean(avg): {_format_float(aggregate['delta_mean_mean'])}",
            f"peak_delta(max): {_format_float(aggregate['peak_delta_hz_max'])} Hz",
        ]))
    fig.suptitle('\n'.join(suptitle_lines))
    _tight_layout_figure(fig, top=0.93, reserve_right=True)

    if save_dir is not None:
        fig_filename = Path(save_dir) / f"singleUnit_psth_raster_all_events_seperated_probe{probeLetter}_clusterID_{int(cluster_id)}.png"
        fig_filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_filename, bbox_inches='tight')

    return fig, ax

def _infer_epoch_color(epoch_label):
    label = str(epoch_label).lower()
    if 'baseline' in label:
        return 'green'
    if 'stim' in label:
        return 'blue'
    if 'wash' in label:
        return 'red'
    return 'black'


def _build_epoch_event_map(event_times, event_meta=None, epoch_event_times_map=None, epoch_column='condition_epoch'):
    if epoch_event_times_map is not None:
        ordered_map = []
        for epoch_label, epoch_times in epoch_event_times_map.items():
            ordered_map.append((str(epoch_label), np.asarray(epoch_times)))
        return ordered_map

    if event_meta is None:
        raise ValueError('Provide event_meta or epoch_event_times_map.')
    if epoch_column not in event_meta.columns:
        raise ValueError(f'{epoch_column} was not found in event_meta.')
    if 'start_time' not in event_meta.columns:
        raise ValueError('event_meta must contain a start_time column.')

    event_times = np.asarray(event_times, dtype=float)
    meta = event_meta.copy()
    meta = meta.dropna(subset=['start_time', epoch_column]).copy()

    if 'trial_index0' in meta.columns:
        meta = meta.sort_values('trial_index0').reset_index(drop=True)
    else:
        meta = meta.sort_values('start_time').reset_index(drop=True)

    # If the metadata rows already correspond 1:1 with the provided event_times,
    # preserve that trial order instead of rematching by start_time.
    if len(meta) == len(event_times):
        meta = meta.copy()
        meta['_event_time_assigned'] = event_times

        ordered_map = []
        used_labels = []
        for epoch_label, epoch_df in meta.groupby(epoch_column, sort=False):
            label = str(epoch_label)
            if label in used_labels:
                continue
            used_labels.append(label)
            epoch_times_group = epoch_df['_event_time_assigned'].astype(float).to_numpy()
            ordered_map.append((label, epoch_times_group))

        return ordered_map

    meta['_start_time_key'] = np.round(meta['start_time'].astype(float), 6)
    event_key_order = np.round(event_times, 6)
    meta = meta[meta['_start_time_key'].isin(event_key_order)].copy()
    if meta.empty:
        raise ValueError('No matching event_meta rows were found for the provided event_times.')

    ordered_map = []
    used_labels = []
    for epoch_label, epoch_df in meta.groupby(epoch_column, sort=False):
        label = str(epoch_label)
        if label in used_labels:
            continue
        used_labels.append(label)
        epoch_times_group = epoch_df['start_time'].astype(float).to_numpy()
        ordered_map.append((label, epoch_times_group))

    return ordered_map


def singleUnit_psth_raster_epoch_stacked(df_units, probeLetter, cluster_id, event_times, pre, post, bin_size=0.01, event_name=None, event_meta=None, epoch_event_times_map=None, epoch_column='condition_epoch', namespace=None, smooth=True, var_bars=True, smooth_window=5, show_data_in_title=True):
    if namespace is None:
        namespace = globals()

    metadata = _collect_unit_plot_metadata(df_units, probeLetter, cluster_id)
    spike_times = metadata['spike_times']
    event_times = np.asarray(event_times)

    if event_name is None:
        event_name = _get_event_label(namespace, event_times, 'event_times')

    epoch_event_map = _build_epoch_event_map(
        event_times=event_times,
        event_meta=event_meta,
        epoch_event_times_map=epoch_event_times_map,
        epoch_column=epoch_column,
    )

    n_rows = len(epoch_event_map) + 1
    fig, ax = plt.subplots(n_rows, 1, figsize=(12, max(8, 2.6 * n_rows)), sharex=True, squeeze=False)
    ax = ax[:, 0]

    epoch_summaries = []
    total_events = 0
    raster_trial_offset = 0

    for row_idx, (epoch_label, epoch_times) in enumerate(epoch_event_map):
        epoch_times = np.asarray(epoch_times)
        epoch_color = _infer_epoch_color(epoch_label)
        total_events += int(len(epoch_times))

        ax[row_idx].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax[row_idx].set_xlim(-pre, post)
        ax[row_idx].set_ylabel('Firing Rate [Hz]')
        ax[row_idx].set_xlabel('Time [s]')

        if len(epoch_times) == 0:
            ax[row_idx].set_title('\n'.join([epoch_label, 'no events']))
            continue

        psth, var, edges, bytrial = trial_by_trial(spike_times, epoch_times, pre, post, bin_size)
        stats = _summarize_trial_response(psth, edges, bytrial, pre, bin_size)
        stats['n_events'] = int(len(epoch_times))
        stats['epoch_label'] = epoch_label
        epoch_summaries.append(stats)

        _plot_psth(ax[row_idx], edges, psth, var, stats['mean_baseline'], color=epoch_color, smooth=smooth, var_bars=var_bars, smooth_window=smooth_window)

        detail_lines = [
            _join_title_parts([
                f"n_events: {stats['n_events']}",
                f"n_trials: {stats['trial_count']}",
                f"mean_baseline: {_format_float(stats['mean_baseline'])}",
                f"std_baseline: {_format_float(stats['std_baseline'])}",
            ]),
            _join_title_parts([
                f"mean_response: {_format_float(stats['mean_response'])}",
                f"delta_mean: {_format_float(stats['delta_mean'])}",
                f"trials_zMean: {_format_float(stats['trial_z_mean'])}",
                f"peak_delta: {_format_float(stats['peak_delta_hz'])} Hz @ {_format_float(stats['peak_time_s'])} s",
            ]),
        ]
        title_lines = [epoch_label]
        if show_data_in_title:
            title_lines.extend(detail_lines)
        ax[row_idx].set_title('\n'.join(title_lines))

        for t, time in enumerate(epoch_times):
            trial_spikes = spike_times[(spike_times > time - pre) & (spike_times < time + post)]
            trial_spikes = trial_spikes - time
            ax[-1].scatter(trial_spikes, [raster_trial_offset + t] * len(trial_spikes), marker='|', color=epoch_color, s=4, alpha=0.6)

        raster_trial_offset += len(epoch_times)

    ax[-1].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax[-1].set_xlim(-pre, post)
    ax[-1].set_ylabel('Trial')
    ax[-1].set_xlabel('Time [s]')

    epoch_count_parts = [f'{epoch_label}: {len(epoch_times)}' for epoch_label, epoch_times in epoch_event_map]
    ax[-1].set_title('\n'.join([
        _join_title_parts([event_name, 'Raster Plot', f'total_events: {len(event_times)}', f'total_trials: {raster_trial_offset}']),
        _join_title_parts(epoch_count_parts),
    ]))

    aggregate = _aggregate_summary(epoch_summaries)
    epoch_order_line = _join_title_parts([epoch_label for epoch_label, _ in epoch_event_map])
    suptitle_lines = _build_unit_header_lines(
        metadata,
        pre,
        post,
        bin_size,
        extra_parts=[
            f'event: {event_name}',
            f'epoch_column: {epoch_column}',
            f'n_epochs: {len(epoch_event_map)}',
            f'total_events: {len(event_times)}',
            f"smooth: {'yes' if smooth else 'no'}",
            f'smooth_window: {smooth_window}',
        ],
    )
    suptitle_lines.append(epoch_order_line)
    if aggregate:
        suptitle_lines.append(_join_title_parts([
            f"mean_baseline(avg): {_format_float(aggregate['mean_baseline_mean'])}",
            f"mean_response(avg): {_format_float(aggregate['mean_response_mean'])}",
            f"delta_mean(avg): {_format_float(aggregate['delta_mean_mean'])}",
            f"trials_zMean(avg): {_format_float(aggregate['trial_z_mean_mean'])}",
            f"peak_delta(max): {_format_float(aggregate['peak_delta_hz_max'])} Hz",
        ]))
    fig.suptitle('\n'.join(suptitle_lines))
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, ax


def singleUnit_psth_raster_epoch_gradient(df_units, probeLetter, cluster_id, event_times, pre, post, bin_size=0.01, event_name=None, event_meta=None, epoch_event_times_map=None, epoch_column='condition_epoch', namespace=None, smooth=True, var_bars=True, smooth_window=5, show_data_in_title=True,save_dir=None):
    if namespace is None:
        namespace = globals()

    metadata = _collect_unit_plot_metadata(df_units, probeLetter, cluster_id)
    spike_times = metadata['spike_times']
    event_times = np.asarray(event_times)

    if event_name is None:
        event_name = _get_event_label(namespace, event_times, 'event_times')

    epoch_event_map = _build_epoch_event_map(
        event_times=event_times,
        event_meta=event_meta,
        epoch_event_times_map=epoch_event_times_map,
        epoch_column=epoch_column,
    )

    baseline_items = []
    stim_items = []
    wash_items = []
    for epoch_label, epoch_times in epoch_event_map:
        label_lower = str(epoch_label).lower()
        entry = (str(epoch_label), np.asarray(epoch_times, dtype=float))
        if 'stim' in label_lower:
            stim_items.append(entry)
        elif 'wash' in label_lower:
            wash_items.append(entry)
        else:
            baseline_items.append(entry)

    stim_colors = _gradient_colors('#1f77b4', len(stim_items), light_mix=0.7)
    wash_colors = _gradient_colors('#d62728', len(wash_items), light_mix=0.7)
    baseline_colors = ['#2ca02c'] * len(baseline_items)

    fig, ax = plt.subplots(3, 1, figsize=(14, 12), sharex=True, squeeze=False)
    ax = ax[:, 0]

    raster_trial_offset = 0
    raster_handles = []
    for entries, colors in ((baseline_items, baseline_colors), (stim_items, stim_colors), (wash_items, wash_colors)):
        for (epoch_label, epoch_times), color in zip(entries, colors):
            if len(epoch_times) == 0:
                continue
            for t, time in enumerate(epoch_times):
                trial_spikes = spike_times[(spike_times > time - pre) & (spike_times < time + post)] - time
                ax[0].scatter(trial_spikes, [raster_trial_offset + t] * len(trial_spikes), marker='|', color=color, s=3, alpha=0.6)
            raster_handles.append(mpatches.Patch(color=color, label=f'{epoch_label} (n={len(epoch_times)})'))
            raster_trial_offset += len(epoch_times)

    ax[0].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax[0].set_xlim(-pre, post)
    ax[0].set_ylabel('Trial')
    ax[0].set_xlabel('Time [s]')
    raster_title = [f'{event_name} | Raster Plot', f'total_events: {len(event_times)}', f'total_trials: {raster_trial_offset}']
    ax[0].set_title(_join_title_parts(raster_title))
    if raster_handles:
        _legend_outside(ax[0], raster_handles)

    def _overlay_epoch_group(axis, entries, colors, title):
        summaries = []
        handles = []
        axis.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        axis.set_xlim(-pre, post)
        axis.set_ylabel('Firing Rate [Hz]')
        axis.set_xlabel('Time [s]')

        if not entries:
            axis.set_title(f'{title} | no epochs')
            return summaries

        for (epoch_label, epoch_times), color in zip(entries, colors):
            if len(epoch_times) == 0:
                continue
            psth, var, edges, bytrial = trial_by_trial(spike_times, epoch_times, pre, post, bin_size)
            stats = _summarize_trial_response(psth, edges, bytrial, pre, bin_size)
            stats['n_events'] = int(len(epoch_times))
            stats['epoch_label'] = epoch_label
            summaries.append(stats)
            _plot_psth(axis, edges, psth, var, stats['mean_baseline'], color=color, smooth=smooth, var_bars=var_bars, smooth_window=smooth_window)
            handles.append(mpatches.Patch(color=color, label=f'{epoch_label} (n={len(epoch_times)})'))

        title_lines = [title]
        if show_data_in_title and summaries:
            aggregate = _aggregate_summary(summaries)
            if aggregate:
                title_lines.append(_join_title_parts([
                    f"epochs: {len(summaries)}",
                    f"events: {sum(item['n_events'] for item in summaries)}",
                    f"mean_response(avg): {_format_float(aggregate['mean_response_mean'])}",
                    f"delta_mean(avg): {_format_float(aggregate['delta_mean_mean'])}",
                ]))
        axis.set_title('\n'.join(title_lines))
        if handles:
            _legend_outside(axis, handles)
        return summaries

    stim_summaries = _overlay_epoch_group(ax[1], stim_items, stim_colors, 'Stimulation PSTH')
    wash_summaries = _overlay_epoch_group(ax[2], wash_items, wash_colors, 'Washout PSTH')

    suptitle_lines = _build_unit_header_lines(
        metadata,
        pre,
        post,
        bin_size,
        extra_parts=[
            f'event: {event_name}',
            f'epoch_column: {epoch_column}',
            f'stim_epochs: {len(stim_items)}',
            f'wash_epochs: {len(wash_items)}',
            f"smooth: {'yes' if smooth else 'no'}",
            f'var_bars: {"yes" if var_bars else "no"}',
        ],
    )
    fig.suptitle('\n'.join(suptitle_lines))
    _tight_layout_figure(fig, top=0.95, reserve_right=True)

    if save_dir is not None:
        fig_filename = Path(save_dir) / f"singleUnit_psth_raster_epoch_gradient_probe{probeLetter}_clusterID_{int(cluster_id)}.png"
        fig_filename.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_filename, bbox_inches='tight')

    return fig, ax


def allUnits_psth_raster_epoch_gradient(df_units, probeLetter, brain_region, event_times, event_name=None, event_meta=None, epoch_event_times_map=None, epoch_column='condition_epoch', label=False, KSlabel=False, pre=0.5, post=1, bin_size=0.01, save_dir=None, var_bars=True, smooth=True, smooth_window=5, show_data_in_title=True):
    df1 = _select_units(
        df_units,
        probeLetter=probeLetter,
        brain_region=brain_region,
        label=label,
        KSlabel=KSlabel,
    )
    saved_paths = []
    for unit_id in _iter_unit_ids(df1):
        fig, ax = singleUnit_psth_raster_epoch_gradient(
            df_units=df1,
            probeLetter=probeLetter,
            cluster_id=int(unit_id),
            event_times=event_times,
            event_name=event_name,
            event_meta=event_meta,
            epoch_event_times_map=epoch_event_times_map,
            epoch_column=epoch_column,
            pre=pre,
            post=post,
            smooth=smooth,
            var_bars=var_bars,
            smooth_window=smooth_window,
            show_data_in_title=show_data_in_title,
        )
        row = _find_unit_row(df1, unit_id)
        region = _unit_region(row)
        saved_paths.append(_save_figure(fig, save_dir, f'clusterID_{int(unit_id)}_probe{probeLetter}_{region}.png'))
        if save_dir is not None:
            _release_figure(fig)
    return saved_paths


def allUnits_psth_raster_subplots_stim_seperated(df_units, probeLetter, brain_region, event_name_subplots=None, trail_indices=None, pre=None, post=None, bin_size=0.01, event_names=None, namespace=None, smooth=True, var_bars=True, smooth_window=5, show_data_in_title=True, save_dir=None, event_times_subplots=None, label=False, KSlabel=False):
    df1 = _select_units(
        df_units,
        probeLetter=probeLetter,
        brain_region=brain_region,
        label=label,
        KSlabel=KSlabel,
    )
    saved_paths = []
    for unit_id in _iter_unit_ids(df1):
        fig, ax = singleUnit_psth_raster_subplots_stim_seperated(
            df_units=df1,
            probeLetter=probeLetter,
            cluster_id=int(unit_id),
            event_name_subplots=event_name_subplots,
            trail_indices=trail_indices,
            pre=pre,
            post=post,
            bin_size=bin_size,
            event_names=event_names,
            namespace=namespace,
            smooth=smooth,
            var_bars=var_bars,
            smooth_window=smooth_window,
            show_data_in_title=show_data_in_title,
            event_times_subplots=event_times_subplots,
        )
        row = _find_unit_row(df1, unit_id)
        region = _unit_region(row)
        saved_paths.append(_save_figure(fig, save_dir, f'clusterID_{int(unit_id)}_probe{probeLetter}_{region}.png'))
        if save_dir is not None:
            _release_figure(fig)
    return saved_paths


def allUnits_psth_raster_2(df_units, df_stim, brain_region=None, title_name='Not Set', event_times=None, label=False, KSlabel=False, all_units=True, dot_size=0.5, pre=0.5, post=1, bin_size=0.05, epoch1='pellet_delivery_timestamp', probeLetter=None, save_dir=None, var_bars=True):
    event_times = _resolve_event_times(event_times=event_times, df_stim=df_stim, epoch1=epoch1)
    df1 = _select_units(df_units, probeLetter=probeLetter, brain_region=brain_region, label=label, KSlabel=KSlabel, all_units=all_units)
    saved_paths = []
    for unit_id in _iter_unit_ids(df1):
        event_label = title_name if title_name != 'Not Set' else epoch1
        fig, ax = singleUnit_psth_raster_test(
            df_units=df1,
            probeLetter=probeLetter,
            cluster_id=int(unit_id),
            event_times=event_times,
            event_label=event_label,
            pre=pre,
            post=post,
            bin_size=bin_size,
            namespace={event_label: event_times},
            var_bars=var_bars,
        )
        row = _find_unit_row(df1, unit_id)
        region = _unit_region(row)
        saved_paths.append(_save_figure(fig, save_dir, f'clusterID_{int(unit_id)}_probe{probeLetter}_{region}.png'))
        if save_dir is not None:
            _release_figure(fig)
    return saved_paths


def psth_raster_stim_seperated_baseline(df_units, df_stim, brain_region=None, event_name='Not Set', trail_indices=None, event_times=None, label=False, KSlabel_good=False, kslabel_mua=False, all_units=True, dot_size=0.5, pre=-1.5, post=1.5, bin_size=0.05, epoch1='pellet_delivery_timestamp', probeLetter=None, save_dir=None, var_bars=True):
    return psth_raster_stim_seperated(
        df_units=df_units,
        df_stim=df_stim,
        brain_region=brain_region,
        event_name=event_name,
        trail_indices=trail_indices,
        event_times=event_times,
        label=label,
        KSlabel=KSlabel_good,
        all_units=all_units,
        dot_size=dot_size,
        pre=abs(pre),
        post=abs(post),
        bin_size=bin_size,
        epoch1=epoch1,
        probeLetter=probeLetter,
        save_dir=save_dir,
        kslabel_mua=kslabel_mua,
        var_bars=var_bars,
    )


def psth_raster_stim_seperated(df_units, df_stim, brain_region, event_name='Not Set', trail_indices=None, event_times=None, label=False, KSlabel=False, all_units=True, dot_size=0.5, pre=0.5, post=1, bin_size=0.05, epoch1='pellet_delivery_timestamp', probeLetter=None, save_dir=None, kslabel_mua=False, var_bars=True):
    if trail_indices is None:
        raise ValueError('trail_indices is required.')
    event_times = _resolve_event_times(event_times=event_times, df_stim=df_stim, epoch1=epoch1)
    df1 = _select_units(
        df_units,
        probeLetter=probeLetter,
        brain_region=brain_region,
        label=label,
        KSlabel=KSlabel,
        kslabel_mua=kslabel_mua,
        all_units=all_units,
    )
    saved_paths = []
    for unit_id in _iter_unit_ids(df1):
        row = _find_unit_row(df1, unit_id)
        spike_times = np.asarray(row['spike_times'], dtype=float)
        region = _unit_region(row)
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle(f'{event_name} | cluster_id: {int(unit_id)} | probe: {probeLetter} | region: {region}')
        max_y = _plot_grouped_overlay(ax[0], ax[1], spike_times, event_times, trail_indices, abs(pre), abs(post), bin_size, var_bars=var_bars, dot_size=max(0.5, dot_size * 6))
        ax[0].set_title('PSTH')
        ax[0].set_ylabel('Firing Rate [Hz]')
        ax[1].set_title('Raster Plot')
        ax[1].set_ylabel('Trial')
        ax[1].set_xlabel('Time (s)')
        ax[0].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax[1].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax[0].set_xlim(-abs(pre), abs(post))
        ax[1].set_xlim(-abs(pre), abs(post))
        if max_y >= 0:
            ax[1].set_ylim(-1, max_y + 1)
        _tight_layout_figure(fig, top=0.95, reserve_right=True)
        saved_paths.append(_save_figure(fig, save_dir, f'clusterID_{int(unit_id)}_probe{probeLetter}_{region}.png'))
        if save_dir is not None:
            _release_figure(fig)
    return saved_paths


def psth_raster_all_events_seperated(df_units, df_stim, brain_region, event_name='Not Set', variables=None, trail_indices=None, label=False, KSlabel=False, all_units=True, dot_size=0.5, pre=0.5, post=1, bin_size=0.05, epoch1='pellet_delivery_timestamp', probeLetter=None, save_dir=None, var_bars=True):
    if not variables:
        raise ValueError('variables is required.')
    if trail_indices is None:
        raise ValueError('trail_indices is required.')
    df1 = _select_units(df_units, probeLetter=probeLetter, brain_region=brain_region, label=label, KSlabel=KSlabel, all_units=all_units)
    saved_paths = []
    legend_handles = _grouped_overlay_legend_handles(trail_indices)
    for unit_id in _iter_unit_ids(df1):
        row = _find_unit_row(df1, unit_id)
        spike_times = np.asarray(row['spike_times'], dtype=float)
        region = _unit_region(row)
        fig, ax = _multi_event_columns(len(variables), width_per_col=8.5, fig_height=12.0)
        fig.suptitle(f'cluster_id: {int(unit_id)} | probe: {probeLetter} | region: {region}')
        for col, (event_label, event_values) in enumerate(variables.items()):
            event_times_panel = np.asarray(event_values, dtype=float)
            max_y = _plot_grouped_overlay(
                ax[0, col],
                ax[1, col],
                spike_times,
                event_times_panel,
                trail_indices,
                abs(pre),
                abs(post),
                bin_size,
                var_bars=var_bars,
                dot_size=max(0.5, dot_size * 6),
                show_legend=False,
            )
            ax[0, col].set_title(event_label)
            ax[0, col].set_ylabel('Firing Rate [Hz]')
            ax[1, col].set_title(event_label)
            ax[1, col].set_ylabel('Trial')
            ax[1, col].set_xlabel('Time (s)')
            ax[0, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax[1, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax[0, col].set_xlim(-abs(pre), abs(post))
            ax[1, col].set_xlim(-abs(pre), abs(post))
            if max_y >= 0:
                ax[1, col].set_ylim(-1, max_y + 1)
        if legend_handles:
            fig.legend(
                handles=legend_handles,
                loc='center left',
                bbox_to_anchor=(0.82, 0.5),
                frameon=False,
                title='Trial group',
            )
        _tight_layout_figure(fig, top=0.95, reserve_right=True)
        saved_paths.append(_save_figure(fig, save_dir, f'clusterID_{int(unit_id)}_probe{probeLetter}_{region}.png'))
        if save_dir is not None:
            _release_figure(fig)
    return saved_paths


def psth_raster_all_events_seperated_plus_opto(df_units, df_stim, brain_region, event_name='Not Set', variables=None, var_bars=True, trail_indices=None, label=False, KSlabel=False, all_units=True, dot_size=0.5, pre=0.5, post=1, bin_size=0.05, epoch1='pellet_delivery_timestamp', probeLetter=None, save_dir=None):
    if not variables:
        raise ValueError('variables is required.')
    if trail_indices is None:
        raise ValueError('trail_indices is required.')
    df1 = _select_units(df_units, probeLetter=probeLetter, brain_region=brain_region, label=label, KSlabel=KSlabel, all_units=all_units)
    saved_paths = []
    for unit_id in _iter_unit_ids(df1):
        row = _find_unit_row(df1, unit_id)
        spike_times = np.asarray(row['spike_times'], dtype=float)
        region = _unit_region(row)
        fig, ax = _multi_event_columns(len(variables))
        fig.suptitle(f'cluster_id: {int(unit_id)} | probe: {probeLetter} | region: {region}')
        for col, (event_label, event_values) in enumerate(variables.items()):
            event_times_panel = np.asarray(event_values, dtype=float)
            if event_label in SPECIAL_UNSPLIT_EVENTS:
                psth, var, edges, bytrial = trial_by_trial(spike_times, event_times_panel, abs(pre), abs(post), bin_size)
                mean_baseline = _baseline_rate(bytrial, abs(pre), bin_size)
                _plot_psth(ax[0, col], edges, psth, var, mean_baseline, color='black', smooth=True, var_bars=var_bars,smooth_window=5)
                _plot_raster_points(ax[1, col], spike_times, event_times_panel, abs(pre), abs(post), color='black', dot_size=max(0.5, dot_size * 6))
            else:
                _plot_grouped_overlay(ax[0, col], ax[1, col], spike_times, event_times_panel, trail_indices, abs(pre), abs(post), bin_size, var_bars=var_bars, dot_size=max(0.5, dot_size * 6))
            ax[0, col].set_title(event_label)
            ax[0, col].set_ylabel('Firing Rate [Hz]')
            ax[1, col].set_title(event_label)
            ax[1, col].set_ylabel('Trial')
            ax[1, col].set_xlabel('Time (s)')
            ax[0, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax[1, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax[0, col].set_xlim(-abs(pre), abs(post))
            ax[1, col].set_xlim(-abs(pre), abs(post))
        _tight_layout_figure(fig, top=0.95, reserve_right=True)
        saved_paths.append(_save_figure(fig, save_dir, f'clusterID_{int(unit_id)}_probe{probeLetter}_{region}.png'))
        if save_dir is not None:
            _release_figure(fig)
    return saved_paths


def psth_raster_all_events_seperated_by_color(df_units, df_stim, brain_region, event_name='Not Set', variables=None, trail_indices=None, label=False, KSlabel=False, all_units=True, dot_size=0.5, pre=0.5, post=1, bin_size=0.05, epoch1='pellet_delivery_timestamp', var_bars=True, probeLetter=None, save_dir=None, gap_threshold=5):
    if not variables:
        raise ValueError('variables is required.')
    if trail_indices is None:
        raise ValueError('trail_indices is required.')
    df1 = _select_units(df_units, probeLetter=probeLetter, brain_region=brain_region, label=label, KSlabel=KSlabel, all_units=all_units)
    color_palette = ['blue', 'red', 'gold', 'green', 'purple', 'cyan', 'magenta', 'orange', 'pink']
    saved_paths = []
    for unit_id in _iter_unit_ids(df1):
        row = _find_unit_row(df1, unit_id)
        spike_times = np.asarray(row['spike_times'], dtype=float)
        region = _unit_region(row)
        fig, ax = _multi_event_columns(len(variables))
        fig.suptitle(f'cluster_id: {int(unit_id)} | probe: {probeLetter} | region: {region}')
        for col, (event_label, event_values) in enumerate(variables.items()):
            event_times_panel = np.asarray(event_values, dtype=float)
            if event_label in SPECIAL_UNSPLIT_EVENTS:
                psth, var, edges, bytrial = trial_by_trial(spike_times, event_times_panel, abs(pre), abs(post), bin_size)
                mean_baseline = _baseline_rate(bytrial, abs(pre), bin_size)
                _plot_psth(ax[0, col], edges, psth, var, mean_baseline, color='black', smooth=True,var_bars=var_bars, smooth_window=5)
                _plot_raster_points(ax[1, col], spike_times, event_times_panel, abs(pre), abs(post), color='black', dot_size=max(0.5, dot_size * 6))
            else:
                legend_handles = []
                for seg_idx, (trial_type, segment) in enumerate(_segment_event_indices(trail_indices, gap_threshold=gap_threshold), start=1):
                    valid = _valid_indices(segment, len(event_times_panel))
                    if len(valid) == 0:
                        continue
                    color = color_palette[(seg_idx - 1) % len(color_palette)]
                    plot_times = event_times_panel[valid]
                    psth, var, edges, bytrial = trial_by_trial(spike_times, plot_times, abs(pre), abs(post), bin_size)
                    mean_baseline = _baseline_rate(bytrial, abs(pre), bin_size)
                    _plot_psth(ax[0, col], edges, psth, var, mean_baseline, color=color, smooth=True, var_bars=var_bars, smooth_window=5)
                    _plot_raster_points(ax[1, col], spike_times, plot_times, abs(pre), abs(post), color=color, dot_size=max(0.5, dot_size * 6), y_values=valid)
                    legend_handles.append(mpatches.Patch(color=color, label=f'{trial_type} segment {seg_idx}'))
                if legend_handles:
                    _legend_outside(ax[0, col], legend_handles)
            ax[0, col].set_title(event_label)
            ax[0, col].set_ylabel('Firing Rate [Hz]')
            ax[1, col].set_title(event_label)
            ax[1, col].set_ylabel('Trial')
            ax[1, col].set_xlabel('Time (s)')
            ax[0, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax[1, col].axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
            ax[0, col].set_xlim(-abs(pre), abs(post))
            ax[1, col].set_xlim(-abs(pre), abs(post))
        _tight_layout_figure(fig, top=0.95, reserve_right=True)
        saved_paths.append(_save_figure(fig, save_dir, f'clusterID_{int(unit_id)}_probe{probeLetter}_{region}.png'))
        if save_dir is not None:
            _release_figure(fig)
    return saved_paths


def allUnits_psth_raster_epoch(df_units, df_stim, probeLetter, brain_region, label=False, KSlabel=False, dot_size=0.5, pre=0.5, post=1, binSizeRaster=0.05, epoch1='pellet_delivery_timestamp', save_dir=None, var_bars=True):
    return allUnits_psth_raster_figures(
        df_units=df_units,
        df_stim=df_stim,
        probeLetter=probeLetter,
        brain_region=brain_region,
        label=label,
        KSlabel=KSlabel,
        pre=pre,
        post=post,
        binSizeRaster=binSizeRaster,
        epoch1=epoch1,
        save_dir=save_dir,
        var_bars=var_bars,
    )


def allUnits_psth_raster_select_eventTimes(df_units, df_stim, probeLetter, brain_region, event_times_to_use=None, label=False, KSlabel=False, dot_size=0.5, pre=0.5, post=1, binSizeRaster=0.05, epoch1='pellet_delivery_timestamp', save_dir=None, var_bars=True):
    return allUnits_psth_raster_2(
        df_units=df_units,
        df_stim=df_stim,
        brain_region=brain_region,
        title_name='selected_event_times',
        event_times=event_times_to_use,
        label=label,
        KSlabel=KSlabel,
        all_units=True,
        dot_size=dot_size,
        pre=pre,
        post=post,
        bin_size=binSizeRaster,
        epoch1=epoch1,
        probeLetter=probeLetter,
        save_dir=save_dir,
        var_bars=var_bars,
    )


def allUnits_psth_raster_figures(df_units, df_stim, probeLetter, brain_region, label=False, KSlabel=False, pre=0.5, post=1, binSizeRaster=0.05, epoch1='pellet_detected_timestamp', save_dir=None, var_bars=True):
    return allUnits_psth_raster_2(
        df_units=df_units,
        df_stim=df_stim,
        brain_region=brain_region,
        title_name=epoch1,
        event_times=None,
        label=label,
        KSlabel=KSlabel,
        all_units=True,
        pre=pre,
        post=post,
        bin_size=binSizeRaster,
        epoch1=epoch1,
        probeLetter=probeLetter,
        save_dir=save_dir,
        var_bars=var_bars,
    )


def multiRegion_raster_figures(df_units, df_stim, brain_regions, probe_units, probe_letters=['A', 'B', 'C', 'D'], label=False, KSlabel=False, pre=0.5, post=1, binSizeRaster=0.05, epoch1='pellet_delivery_timestamp', save_dir=None, highlight_time_zero=False, dot_size=0.12, background_colors=None):
    event_times = _resolve_event_times(df_stim=df_stim, epoch1=epoch1)
    if background_colors is None:
        background_colors = DEFAULT_REGION_BACKGROUND_COLORS
    fig, axs = plt.subplots(len(probe_letters), 1, figsize=(14, max(3.5, len(probe_letters)) * 3.2), sharex=True)
    axs = np.atleast_1d(axs)
    fig.suptitle(f'Aligned to {epoch1}; Units {probe_units}')
    for idx, (probe, unit_id, region, ax) in enumerate(zip(probe_letters, probe_units, brain_regions, axs)):
        df_probe = _select_units(df_units, probeLetter=probe, label=label, KSlabel=KSlabel)
        row = _find_unit_row(df_probe, unit_id)
        spike_times = np.asarray(row['spike_times'], dtype=float)
        ax.set_facecolor(background_colors[idx % len(background_colors)])
        _plot_raster_points(ax, spike_times, event_times, abs(pre), abs(post), color='black', dot_size=max(0.5, dot_size * 8))
        if highlight_time_zero:
            ax.axvspan(0, 0.001, color='gray', alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(str(region), rotation=0, labelpad=30, va='center')
        ax.set_title(f'Region: {region} | cluster_id: {int(unit_id)}')
        ax.set_xlim(-abs(pre), abs(post))
    axs[-1].set_xlabel('Time (s)')
    _tight_layout_figure(fig, top=0.95, reserve_right=False)
    saved = _save_figure(fig, save_dir, f'multi_region_raster_units_{list(probe_units)}.png')
    if save_dir is not None:
        _release_figure(fig)
    return saved


def _multi_region_raster_psth_common(df_units, event_times, brain_regions, probe_units, probe_letters, label, KSlabel, pre, post, binSizeRaster, binSizePSTH, save_dir, highlight_time_zero, normalize_psth, dot_size, background_colors, title):
    if background_colors is None:
        background_colors = DEFAULT_REGION_BACKGROUND_COLORS
    fig, axs = plt.subplots(len(probe_letters) + 1, 1, figsize=(16, max(4, len(probe_letters) + 1) * 3.2), sharex=True)
    axs = np.atleast_1d(axs)
    psth_ax = axs[0]
    fig.suptitle(f'Aligned to {title}; Units {probe_units}')
    for idx, (probe, unit_id, region, ax) in enumerate(zip(probe_letters, probe_units, brain_regions, axs[1:])):
        df_probe = _select_units(df_units, probeLetter=probe, label=label, KSlabel=KSlabel)
        row = _find_unit_row(df_probe, unit_id)
        spike_times = np.asarray(row['spike_times'], dtype=float)
        psth, _, edges, _ = trial_by_trial(spike_times, event_times, abs(pre), abs(post), binSizePSTH)
        if normalize_psth and np.nanmax(np.abs(psth)) > 0:
            psth = psth / np.nanmax(np.abs(psth))
        ax.set_facecolor(background_colors[idx % len(background_colors)])
        max_y = _plot_raster_points(ax, spike_times, event_times, abs(pre), abs(post), color='black', dot_size=max(0.5, dot_size))
        if max_y >= 0:
            ax.set_ylim(-1, max_y + 1)
        if highlight_time_zero:
            ax.axvspan(0, 0.001, color='gray', alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.set_ylabel(str(region), rotation=0, labelpad=30, va='center')
        vibrant_color = make_color_vibrant(background_colors[idx % len(background_colors)])
        psth_ax.plot(edges, psth, label=str(region), color=vibrant_color, linewidth=2)
        ax.set_xlim(-abs(pre), abs(post))
    psth_ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    psth_ax.set_title('PSTH - All Brain Regions')
    psth_ax.set_ylabel('Firing Rate (normalized)' if normalize_psth else 'Firing Rate')
    legend_handles = _labeled_line_patches(psth_ax)
    if legend_handles:
        _legend_outside(psth_ax, legend_handles)
    axs[-1].set_xlabel('Time (s)')
    _tight_layout_figure(fig, top=0.95, reserve_right=True)
    saved = _save_figure(fig, save_dir, f'multi_region_psth_raster_units_{list(probe_units)}.png')
    if save_dir is not None:
        _release_figure(fig)
    return saved


def multiRegion_raster_psth_figures(df_units, df_stim, brain_regions, probe_units, probe_letters=['A', 'B', 'C', 'D'], label=False, KSlabel=False, pre=0.5, post=1, binSizeRaster=0.05, binSizePSTH=0.025, epoch1='pellet_detected_timestamp', save_dir=None, highlight_time_zero=False, normalize_psth=False, dot_size=0.8, background_colors=None):
    event_times = _resolve_event_times(df_stim=df_stim, epoch1=epoch1)
    return _multi_region_raster_psth_common(
        df_units=df_units,
        event_times=event_times,
        brain_regions=brain_regions,
        probe_units=probe_units,
        probe_letters=probe_letters,
        label=label,
        KSlabel=KSlabel,
        pre=pre,
        post=post,
        binSizeRaster=binSizeRaster,
        binSizePSTH=binSizePSTH,
        save_dir=save_dir,
        highlight_time_zero=highlight_time_zero,
        normalize_psth=normalize_psth,
        dot_size=dot_size,
        background_colors=background_colors,
        title=epoch1,
    )


def multiRegion_raster_psth_normalized(df_units, df_stim, brain_regions, probe_units, probe_letters=['A', 'B', 'C', 'D'], label=False, KSlabel=False, pre=0.5, post=1, binSizeRaster=0.05, binSizePSTH=0.025, event_times=None, epoch1='pellet_detected_timestamp', save_dir=None, highlight_time_zero=False, normalize_psth=False, dot_size=0.8, background_colors=None):
    event_times = _resolve_event_times(event_times=event_times, df_stim=df_stim, epoch1=epoch1)
    return _multi_region_raster_psth_common(
        df_units=df_units,
        event_times=event_times,
        brain_regions=brain_regions,
        probe_units=probe_units,
        probe_letters=probe_letters,
        label=label,
        KSlabel=KSlabel,
        pre=pre,
        post=post,
        binSizeRaster=binSizeRaster,
        binSizePSTH=binSizePSTH,
        save_dir=save_dir,
        highlight_time_zero=highlight_time_zero,
        normalize_psth=True if not normalize_psth else normalize_psth,
        dot_size=dot_size,
        background_colors=background_colors,
        title='custom_event_times',
    )


def probe_units_heatmap(df_units, df_stim, probeLetter, selected_units=None, pre=0.5, post=1, bin_size=0.05, epoch1='pellet_detected_timestamp', label=False, KSlabel=True, save_dir=None, max_fr=60, show_unit_labels=True, normalize_fr=False, event_times=None):
    event_times = _resolve_event_times(event_times=event_times, df_stim=df_stim, epoch1=epoch1)
    df_probe = _select_units(df_units, probeLetter=probeLetter, label=label, KSlabel=KSlabel)
    unit_ids = _iter_unit_ids(df_probe, selected_units=selected_units)
    heatmap_data, labels, centers = _heatmap_rows(df_probe, unit_ids, event_times, abs(pre), abs(post), bin_size, normalize_fr=normalize_fr, max_fr=max_fr)
    if heatmap_data.size == 0:
        raise ValueError(f'No units available for probe {probeLetter}.')
    fig, ax = plt.subplots(figsize=(16, max(3, len(labels) * 0.25)))
    vmin, vmax = (-2, 2) if normalize_fr else (0, max_fr)
    sns.heatmap(heatmap_data, cmap='RdBu_r' if normalize_fr else 'jet', ax=ax, cbar=True, xticklabels=False, yticklabels=labels if show_unit_labels else False, vmin=vmin, vmax=vmax, cbar_kws={'shrink': 0.6})
    tick_positions = np.linspace(0, heatmap_data.shape[1] - 1, min(10, heatmap_data.shape[1])).astype(int)
    tick_labels = np.round(np.linspace(-abs(pre), abs(post), len(tick_positions)), 2)
    ax.set_xticks(tick_positions + 0.5)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_title(f'Heatmap for Probe {probeLetter}')
    ax.set_ylabel('Units')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    saved = _save_figure(fig, save_dir, f'probe_{probeLetter}_heatmap.png')
    if save_dir is not None:
        _release_figure(fig)
    return fig, ax, saved


def sorted_heatmap(df_units, df_stim, probeLetter, selected_units=None, pre=0.5, post=1, bin_size=0.05, epoch1='pellet_detected_timestamp', label=False, KSlabel=True, save_dir=None, max_fr=60, show_unit_labels=True, normalize_fr=True, event_times=None):
    event_times = _resolve_event_times(event_times=event_times, df_stim=df_stim, epoch1=epoch1)
    df_probe = _select_units(df_units, probeLetter=probeLetter, label=label, KSlabel=KSlabel)
    unit_ids = _iter_unit_ids(df_probe, selected_units=selected_units)
    heatmap_data, labels, centers = _heatmap_rows(df_probe, unit_ids, event_times, abs(pre), abs(post), bin_size, normalize_fr=normalize_fr, max_fr=max_fr)
    if heatmap_data.size == 0:
        raise ValueError(f'No units available for probe {probeLetter}.')
    order = sort_units_by_firing_rate_change(heatmap_data, centers, abs(pre), abs(post), bin_size, sort_by_time=False, smoothing_window=1.5)
    sorted_heat = heatmap_data[order]
    sorted_labels = labels[order]
    fig, ax = plt.subplots(figsize=(16, max(3, len(sorted_labels) * 0.25)))
    vmin, vmax = (-2, 2) if normalize_fr else (0, max_fr)
    sns.heatmap(sorted_heat, cmap='RdBu_r' if normalize_fr else 'jet', ax=ax, cbar=True, xticklabels=False, yticklabels=sorted_labels if show_unit_labels else False, vmin=vmin, vmax=vmax, cbar_kws={'shrink': 0.6})
    tick_positions = np.linspace(0, sorted_heat.shape[1] - 1, min(10, sorted_heat.shape[1])).astype(int)
    tick_labels = np.round(np.linspace(-abs(pre), abs(post), len(tick_positions)), 2)
    ax.set_xticks(tick_positions + 0.5)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_title(f'Sorted Heatmap for Probe {probeLetter}')
    ax.set_ylabel('Units')
    ax.set_xlabel('Time (s)')
    fig.tight_layout()
    saved = _save_figure(fig, save_dir, f'probe_{probeLetter}_sorted_heatmap.png')
    if save_dir is not None:
        _release_figure(fig)
    return fig, ax, saved


def multi_probe_units_heatmap(df_units, df_stim, probes, selected_units=None, selected_units_by_subplot=None, times_of_events=None, event_name=None, brain_regions=None, brain_region_filters=None, pre=0.5, post=1, bin_size=0.01, epoch1='pellet_detected_timestamp', label=False, KSlabel=True, save_dir=None, max_fr=60, show_unit_labels=True, normalize_fr=False):
    event_times = _resolve_event_times(event_times=times_of_events, df_stim=df_stim, epoch1=epoch1)
    fig, axes = plt.subplots(len(probes), 1, figsize=(22, max(4, len(probes) * 4.8)), squeeze=False)
    axes = axes.flatten()
    for idx, probeLetter in enumerate(probes):
        region_filter = None
        if brain_region_filters is not None:
            region_filter = brain_region_filters[idx]
        df_probe = _select_units(df_units, probeLetter=probeLetter, brain_region=region_filter, label=label, KSlabel=KSlabel)
        if selected_units_by_subplot is not None:
            unit_ids = _iter_unit_ids(df_probe, selected_units=selected_units_by_subplot[idx])
        elif selected_units is not None:
            unit_ids = _iter_unit_ids(df_probe, selected_units=selected_units.get(probeLetter))
        else:
            unit_ids = _iter_unit_ids(df_probe)
        heatmap_data, labels, _ = _heatmap_rows(df_probe, unit_ids, event_times, abs(pre), abs(post), bin_size, normalize_fr=normalize_fr, max_fr=max_fr)
        ax = axes[idx]
        if heatmap_data.size == 0:
            ax.set_axis_off()
            continue
        vmin, vmax = (-2, 2) if normalize_fr else (0, max_fr)
        sns.heatmap(heatmap_data, cmap='RdBu_r' if normalize_fr else 'jet', ax=ax, cbar=True, xticklabels=False, yticklabels=labels if show_unit_labels else False, vmin=vmin, vmax=vmax, cbar_kws={'shrink': 0.6})
        label_text = brain_regions[idx] if brain_regions else (region_filter if region_filter is not None else probeLetter)
        ax.set_ylabel(label_text, rotation=0, labelpad=30, va='center')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Probe {probeLetter} | {label_text}')
    fig.suptitle(f'Aligned to {event_name or epoch1}')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    saved = _save_figure(fig, save_dir, f'multi_probe_heatmap_{event_name or epoch1}_{abs(pre)}_{abs(post)}.png')
    if save_dir is not None:
        _release_figure(fig)
    return fig, axes, saved


def multi_probe_units_heatmap_smoothed(df_units, df_stim, probes, selected_units=None, selected_units_by_subplot=None, times_of_events=None, event_name=None, brain_regions=None, brain_region_filters=None, pre=0.5, post=1, bin_size=0.01, epoch1='pellet_detected_timestamp', label=False, KSlabel=True, save_dir=None, max_fr=60, show_unit_labels=True, normalize_fr=False, sort_by_time=True, reset_unit_count=False, smoothing_window=1,single_probe=False):
    event_times = _resolve_event_times(event_times=times_of_events, df_stim=df_stim, epoch1=epoch1)
    if single_probe:
        fig, axes = plt.subplots(len(probes), 1, figsize=(12, max(4, len(probes) * 4.8)), squeeze=False)
    else:
        fig, axes = plt.subplots(len(probes), 1, figsize=(12, max(4, len(probes) * 4.8)), squeeze=False)
    axes = axes.flatten()
    for idx, probeLetter in enumerate(probes):
        region_filter = None
        if brain_region_filters is not None:
            region_filter = brain_region_filters[idx]
        df_probe = _select_units(df_units, probeLetter=probeLetter, brain_region=region_filter, label=label, KSlabel=KSlabel)
        if selected_units_by_subplot is not None:
            unit_ids = _iter_unit_ids(df_probe, selected_units=selected_units_by_subplot[idx])
        elif selected_units is not None:
            unit_ids = _iter_unit_ids(df_probe, selected_units=selected_units.get(probeLetter))
        else:
            unit_ids = _iter_unit_ids(df_probe)
        heatmap_data, labels, centers = _heatmap_rows(df_probe, unit_ids, event_times, abs(pre), abs(post), bin_size, normalize_fr=normalize_fr, max_fr=max_fr, smoothing_sigma=smoothing_window)
        ax = axes[idx]
        if heatmap_data.size == 0:
            ax.set_axis_off()
            continue
        order = sort_units_by_firing_rate_change(heatmap_data, centers, abs(pre), abs(post), bin_size, sort_by_time=sort_by_time, smoothing_window=smoothing_window)
        sorted_heat = heatmap_data[order]
        sorted_labels = labels[order]
        y_labels = np.arange(1, len(sorted_labels) + 1) if reset_unit_count else sorted_labels
        vmin, vmax = (-2, 2) if normalize_fr else (0, max_fr)
        sns.heatmap(sorted_heat, cmap='RdBu_r' if normalize_fr else 'jet', ax=ax, cbar=True, xticklabels=False, yticklabels=y_labels if show_unit_labels else False, vmin=vmin, vmax=vmax, cbar_kws={'shrink': 0.6})
        label_text = brain_regions[idx] if brain_regions else (region_filter if region_filter is not None else probeLetter)
        ax.set_ylabel(label_text, rotation=0, labelpad=30, va='center')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Probe {probeLetter} | {label_text}')
    fig.suptitle(f'Aligned to {event_name or epoch1}')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    saved = _save_figure(fig, save_dir, f'_{event_name or epoch1}_{abs(pre)}_{abs(post)}.png')
    if save_dir is not None:
        _release_figure(fig)
    return fig, axes, saved


def multi_probe_units_heatmap_smoothed_stim_seperated(df_units, df_stim, probes, selected_units=None, selected_units_by_subplot=None, times_of_events=None, event_name=None, brain_regions=None, brain_region_filters=None, trail_indices=None, pre=0.5, post=1, bin_size=0.01, epoch1='pellet_detected_timestamp', label=False, KSlabel=True, save_dir=None, max_fr=60, show_unit_labels=True, normalize_fr=False, sort_by_time=True, reset_unit_count=False, smoothing_window=1, single_probe=False):
    if trail_indices is None:
        raise ValueError('trail_indices is required for baseline/stimulation/washout separation.')

    event_times = _resolve_event_times(event_times=times_of_events, df_stim=df_stim, epoch1=epoch1)
    event_label = event_name or epoch1
    phase_map = (
        ('baseline', 'baseline'),
        ('optical_stim', 'stimulation'),
        ('no_optical_stim', 'washout'),
    )

    results = {}
    for trial_key, phase_label in phase_map:
        phase_indices = _valid_indices(trail_indices.get(trial_key, []), len(event_times))
        phase_event_times = event_times[phase_indices]
        results[phase_label] = multi_probe_units_heatmap_smoothed(
            df_units=df_units,
            df_stim=df_stim,
            probes=probes,
            selected_units=selected_units,
            selected_units_by_subplot=selected_units_by_subplot,
            times_of_events=phase_event_times,
            event_name=f'{event_label}_{phase_label}',
            brain_regions=brain_regions,
            brain_region_filters=brain_region_filters,
            pre=pre,
            post=post,
            bin_size=bin_size,
            epoch1=epoch1,
            label=label,
            KSlabel=KSlabel,
            save_dir=save_dir,
            max_fr=max_fr,
            show_unit_labels=show_unit_labels,
            normalize_fr=normalize_fr,
            sort_by_time=sort_by_time,
            reset_unit_count=reset_unit_count,
            smoothing_window=smoothing_window,
            single_probe=single_probe,
        )
    return results


__all__ = [
    'combine_merged_units',
    'select_units',
    'split_units_by_probe_and_region',
    'batch_run_by_probe_and_region',
    'flatten_nested_trial_numbers',
    'build_trial_index_groups',
    'build_event_name_subplots',
    'trial_by_trial',
    'smooth_data',
    'sort_units_by_firing_rate_change',
    'singleUnit_psth_raster_test',
    'singleUnit_psth_raster_subplots',
    'singleUnit_psth_raster_subplots_stim_seperated',
    'singleUnit_psth_raster_all_events_seperated',
    'singleUnit_psth_raster_epoch_stacked',
    'singleUnit_psth_raster_epoch_gradient',
    'allUnits_psth_raster_subplots_stim_seperated',
    'allUnits_psth_raster_epoch_gradient',
    'allUnits_psth_raster_2',
    'psth_raster_stim_seperated_baseline',
    'psth_raster_stim_seperated',
    'psth_raster_all_events_seperated',
    'psth_raster_all_events_seperated_plus_opto',
    'psth_raster_all_events_seperated_by_color',
    'allUnits_psth_raster_epoch',
    'allUnits_psth_raster_select_eventTimes',
    'create_save_dir',
    'allUnits_psth_raster_figures',
    'multiRegion_raster_figures',
    'multiRegion_raster_psth_figures',
    'multiRegion_raster_psth_normalized',
    'probe_units_heatmap',
    'sorted_heatmap',
    'multi_probe_units_heatmap',
    'multi_probe_units_heatmap_smoothed',
    'multi_probe_units_heatmap_smoothed_stim_seperated',
]




