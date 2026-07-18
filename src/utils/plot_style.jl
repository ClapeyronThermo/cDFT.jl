"""
Shared default styling constants for `MakiecDFTExt`/`PlotscDFTExt`, ported 1:1 from the
house matplotlib `rcParams` style used for this package's figures. These are plain
constants (fallback defaults), not mutable global state — every `plot(system, ρ; ...)`
method accepts `color_scheme`/`font`/`width`/`dpi`/`grid` kwargs to override them per call.
Defined once here so the two plotting extensions can't drift apart.
"""

# Single/double journal-column widths in inches, at the paper's working DPI. Matches
# `WIDTH = 1.5 * 8.3 / 2.54`, `DOUBLE_WIDTH = 1.5 * 17.1 / 2.54`, `DPI = 150`.
const CDFT_WIDTH_IN = 1.5 * 8.3 / 2.54
const CDFT_DOUBLE_WIDTH_IN = 1.5 * 17.1 / 2.54
const CDFT_DPI = 150

# Matches `axes.labelsize=14`, `xtick/ytick.labelsize=12`, `legend.fontsize=10`,
# `font.size=12`.
const CDFT_AXES_LABELSIZE = 14
const CDFT_TICK_LABELSIZE = 12
const CDFT_LEGEND_FONTSIZE = 10
const CDFT_FONT_SIZE = 12

# Matches `axes.prop_cycle` (Julia-blue / teal-green / gold / crimson / purple / gray).
const CDFT_DEFAULT_COLORS = ["#4063D8", "#009B72", "#FFB400", "#D80032", "#9558B2", "#808080"]

# Matches `grid.color="0"`, `grid.linestyle="-"` -- only applied when a caller opts into
# `grid=true`; both extensions default to `grid=false` (current no-grid look, unchanged).
const CDFT_GRID_COLOR = :black
const CDFT_GRID_LINESTYLE = "-"

"""
    cdft_figure_size(width::Symbol, dpi::Real=CDFT_DPI)

Figure size in pixels for `width ∈ (:single, :double)`, at 4:3 aspect ratio (matches
`figure.figsize = (WIDTH, 3*WIDTH/4)`). Returns `(width_px, height_px)`.
"""
function cdft_figure_size(width::Symbol, dpi::Real=CDFT_DPI)
    width_in = width === :single ? CDFT_WIDTH_IN :
               width === :double ? CDFT_DOUBLE_WIDTH_IN :
               error("width must be :single or :double, got :$width")
    width_px = round(Int, width_in * dpi)
    height_px = round(Int, 3 * width_px / 4)
    return (width_px, height_px)
end

export CDFT_DEFAULT_COLORS, CDFT_DPI, cdft_figure_size
