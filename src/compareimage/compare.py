"""
Core image comparison functionality.

This module contains the main functions for comparing images and
identifying differences between them.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

from compareimage.utils import get_color_name, extract_text_in_region, get_text_change_description

# Try to import pytesseract for text detection
try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    print("Warning: pytesseract module not found. Text detection will be disabled.")
    HAS_PYTESSERACT = False


def compare_images(image1_path, image2_path, labels=("Original Image", "Modified Image"), threshold=30,
                   grouping_enabled=True, group_distance=50):
    """
    Compare two images and identify differences with optional grouping of nearby changes.

    Parameters:
    - image1_path: Path to the first image
    - image2_path: Path to the second image
    - labels: Tuple of labels for the images
    - threshold: Pixel difference threshold (0-255)
    - grouping_enabled: Whether to group nearby changes together
    - group_distance: Maximum distance to consider changes as part of the same group

    Returns:
    - Dictionary containing comparison results
    """
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        raise ValueError("Error loading images. Check file paths.")

    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions.")

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    (score, diff) = ssim(gray1, gray2, full=True)

    diff = (diff * 255).astype(np.uint8)

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    diff_image = image1.copy().astype(np.uint8)
    highlight_overlay = image1.copy().astype(np.uint8)
    side_by_side = np.hstack((image1.copy(), image2.copy()))

    abs_diff = cv2.absdiff(image1, image2)

    mask = np.any(abs_diff > threshold, axis=2).astype(np.uint8) * 255

    diff_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    primary_regions = []

    composite = (image1.copy() * 0.5).astype(np.uint8)

    # Try to use text detection if available
    try:
        if HAS_PYTESSERACT:
            text1 = pytesseract.image_to_data(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB),
                                              output_type=pytesseract.Output.DICT)
            text2 = pytesseract.image_to_data(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB),
                                              output_type=pytesseract.Output.DICT)
            has_text_detection = True
        else:
            has_text_detection = False
            print("Warning: Text detection not available (pytesseract not installed).")
    except Exception as e:
        has_text_detection = False
        print(f"Warning: Text detection failed: {str(e)}. Install pytesseract and configure it properly.")

    change_colors = [
        (255, 0, 0),  # Red
        (0, 255, 0),  # Green
        (0, 0, 255),  # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
        (0, 128, 255),  # Light Blue
        (255, 0, 128)  # Pink
    ]

    # Create a list to store all individual changes before grouping
    all_regions = []

    # First, collect all individual changes
    for i, contour in enumerate(diff_contours):
        x, y, w, h = cv2.boundingRect(contour)

        if w * h < 25:  # Skip very small changes that might be noise
            continue

        center_x, center_y = x + w // 2, y + h // 2

        region1 = image1[y:y + h, x:x + w].copy()
        region2 = image2[y:y + h, x:x + w].copy()

        avg_color1_bgr = np.mean(region1, axis=(0, 1)).astype(int)
        avg_color2_bgr = np.mean(region2, axis=(0, 1)).astype(int)

        color1 = get_color_name(avg_color1_bgr)
        color2 = get_color_name(avg_color2_bgr)

        region_text1 = ""
        region_text2 = ""

        if has_text_detection:
            region_text1 = extract_text_in_region(text1, x, y, w, h)
            region_text2 = extract_text_in_region(text2, x, y, w, h)

        color_changed = color1 != color2
        text_changed = region_text1 != region_text2 and (region_text1 != "" or region_text2 != "")

        region_diff = cv2.absdiff(region1, region2)
        region_mask = np.any(region_diff > threshold, axis=2)

        is_old_brighter = (np.mean(region1, axis=2) > np.mean(region2, axis=2))

        region_composite = (region1.copy() * 0.5).astype(np.uint8)

        for ry in range(h):
            for rx in range(w):
                if region_mask[ry, rx]:
                    if is_old_brighter[ry, rx]:
                        region_composite[ry, rx] = [0, 0, 255]  # Red for removed
                    else:
                        region_composite[ry, rx] = [0, 255, 0]  # Green for added

        composite[y:y + h, x:x + w] = region_composite

        if np.mean(avg_color1_bgr) > np.mean(avg_color2_bgr):
            diff_type = "Removed"
        else:
            diff_type = "Added"

        zoom_region1 = cv2.resize(region1, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        zoom_region2 = cv2.resize(region2, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        region_diff_zoomed = cv2.resize(region_diff, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)

        if text_changed:
            change_type = "Text Change"
            change_description = get_text_change_description(region_text1, region_text2)
        elif color_changed:
            change_type = "Color Change"
            change_description = f"Color changed from {color1} to {color2}"
        else:
            change_type = "Content Change"
            change_description = f"{diff_type} content in this region"

        # Calculate pixel-level differences
        diff_percentage_in_region = np.mean(region_mask) * 100

        # Find edge differences (useful for shape changes)
        edge1 = cv2.Canny(region1, 100, 200)
        edge2 = cv2.Canny(region2, 100, 200)
        edge_diff = cv2.bitwise_xor(edge1, edge2)
        edge_diff_percentage = np.sum(edge_diff > 0) / (w * h) * 100

        # Add more detailed change analysis
        detail_analysis = {
            'diff_percentage_in_region': diff_percentage_in_region,
            'edge_change_percentage': edge_diff_percentage,
            'brightness_change': np.mean(region2) - np.mean(region1),
            'contrast_change': np.std(region2) - np.std(region1)
        }

        # Detailed change type analysis
        if edge_diff_percentage > 20 and abs(detail_analysis['brightness_change']) < 10:
            detail_change_type = "Shape/Structure Change"
        elif abs(detail_analysis['brightness_change']) > 20:
            if detail_analysis['brightness_change'] > 0:
                detail_change_type = "Brightness Increased"
            else:
                detail_change_type = "Brightness Decreased"
        elif abs(detail_analysis['contrast_change']) > 10:
            if detail_analysis['contrast_change'] > 0:
                detail_change_type = "Contrast Increased"
            else:
                detail_change_type = "Contrast Decreased"
        else:
            detail_change_type = "Minor Visual Change"

        # Add more details to the description based on our analysis
        detailed_change_description = change_description
        if "Content Change" in change_type:
            detailed_change_description += f" ({detail_change_type})"

        all_regions.append({
            'id': i + 1,  # Temporary ID, will be updated after grouping
            'position': (x, y, w, h),
            'center': (center_x, center_y),
            'area': w * h,
            'change_type': change_type,
            'change_description': change_description,
            'detailed_change_description': detailed_change_description,
            'detailed_change_type': detail_change_type,
            'detail_analysis': detail_analysis,
            'old_color': color1,
            'new_color': color2,
            'old_text': region_text1,
            'new_text': region_text2,
            'difference_type': diff_type,
            'change_color': None,  # Will be assigned after grouping
            'zoomed_region1': zoom_region1,
            'zoomed_region2': zoom_region2,
            'zoomed_diff': region_diff_zoomed,
            'group_id': None  # Will be assigned during grouping
        })

    # Group changes based on proximity if enabled
    difference_regions = []

    if grouping_enabled and len(all_regions) > 1:
        # Extract centers for clustering
        centers = np.array([region['center'] for region in all_regions])

        # Use DBSCAN clustering to group nearby changes
        clustering = DBSCAN(eps=group_distance, min_samples=1).fit(centers)

        # Get the cluster labels
        labels = clustering.labels_

        # Assign group IDs
        for i, region in enumerate(all_regions):
            region['group_id'] = int(labels[i])

        # Number of groups
        n_groups = len(set(labels)) - (1 if -1 in labels else 0)

        # Process each group
        for group_id in range(n_groups):
            # Get all regions in this group
            group_regions = [r for r in all_regions if r['group_id'] == group_id]

            if not group_regions:
                continue

            # Assign a color for this group
            group_color = change_colors[group_id % len(change_colors)]

            # Determine the bounding box that encompasses all regions in the group
            min_x = min(r['position'][0] for r in group_regions)
            min_y = min(r['position'][1] for r in group_regions)
            max_x = max(r['position'][0] + r['position'][2] for r in group_regions)
            max_y = max(r['position'][1] + r['position'][3] for r in group_regions)

            group_w = max_x - min_x
            group_h = max_y - min_y

            # Create a new region representing the group
            group_region = {
                'id': group_id + 1,
                'position': (min_x, min_y, group_w, group_h),
                'area': group_w * group_h,
                'change_color': group_color,
                'contains': len(group_regions),
                'subregions': group_regions,
                'zoomed_region1': None,  # Will be created when needed
                'zoomed_region2': None  # Will be created when needed
            }

            # Determine the primary change type for this group
            change_types = [r['change_type'] for r in group_regions]
            if "Text Change" in change_types:
                primary_change_type = "Text Change"
            elif "Color Change" in change_types:
                primary_change_type = "Color Change"
            else:
                primary_change_type = "Content Change"

            # Extract text changes if any
            text_changes = [r for r in group_regions if r['change_type'] == "Text Change"]
            combined_old_text = " ".join([r['old_text'] for r in text_changes if r['old_text']])
            combined_new_text = " ".join([r['new_text'] for r in text_changes if r['new_text']])

            # Combine change descriptions
            if primary_change_type == "Text Change" and combined_old_text and combined_new_text:
                change_description = f"Text changed from '{combined_old_text}' to '{combined_new_text}'"
            else:
                # Count the types of changes
                change_counts = {}
                for r in group_regions:
                    change_counts[r['detailed_change_type']] = change_counts.get(r['detailed_change_type'], 0) + 1

                # Find the most common change type
                most_common_type = max(change_counts.items(), key=lambda x: x[1])[0]
                change_description = f"Group of {len(group_regions)} changes, mostly {most_common_type}"

            group_region['change_type'] = primary_change_type
            group_region['change_description'] = change_description
            group_region[
                'detailed_change_description'] = f"Group of {len(group_regions)} related changes: {change_description}"
            group_region['detailed_change_type'] = most_common_type
            group_region['old_text'] = combined_old_text
            group_region['new_text'] = combined_new_text

            # Create composite images for the group
            group_region1 = image1[min_y:max_y, min_x:max_x].copy()
            group_region2 = image2[min_y:max_y, min_x:max_x].copy()

            # Create zoomed versions
            zoom_factor = min(3, 300 / max(group_w, group_h))  # Limit zoom for large regions
            zoom_width = int(group_w * zoom_factor)
            zoom_height = int(group_h * zoom_factor)

            group_region['zoomed_region1'] = cv2.resize(group_region1, (zoom_width, zoom_height),
                                                        interpolation=cv2.INTER_NEAREST)
            group_region['zoomed_region2'] = cv2.resize(group_region2, (zoom_width, zoom_height),
                                                        interpolation=cv2.INTER_NEAREST)
            group_region['zoomed_diff'] = cv2.resize(
                cv2.absdiff(group_region1, group_region2),
                (zoom_width, zoom_height),
                interpolation=cv2.INTER_NEAREST
            )

            # Add the group to our final results
            difference_regions.append(group_region)

            # Draw rectangles for the group
            cv2.rectangle(diff_image, (min_x, min_y), (max_x, max_y), group_color, 2)
            cv2.putText(diff_image, f"#{group_id + 1}", (min_x, min_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, group_color, 2)

            # Also draw on side-by-side
            cv2.rectangle(side_by_side, (min_x, min_y), (min_x + group_w, min_y + group_h), group_color, 2)
            cv2.putText(side_by_side, f"#{group_id + 1}", (min_x, min_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, group_color, 2)

            # Draw on second image in side-by-side
            cv2.rectangle(side_by_side, (min_x + image1.shape[1], min_y),
                          (min_x + group_w + image1.shape[1], min_y + group_h), group_color, 2)
            cv2.putText(side_by_side, f"#{group_id + 1}", (min_x + image1.shape[1], min_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, group_color, 2)

            # Create transparent overlay
            overlay = highlight_overlay.copy()
            cv2.rectangle(overlay, (min_x, min_y), (max_x, max_y), group_color, -1)

            # Apply transparency
            alpha = 0.3
            highlight_overlay = cv2.addWeighted(overlay, alpha, highlight_overlay, 1 - alpha, 0)

            # Add outline to highlighted area
            cv2.rectangle(highlight_overlay, (min_x, min_y), (max_x, max_y), group_color, 2)
            cv2.putText(highlight_overlay, f"#{group_id + 1}", (min_x, min_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, group_color, 2)
    else:
        # No grouping, use original regions
        for i, region in enumerate(all_regions):
            region['id'] = i + 1
            region['change_color'] = change_colors[i % len(change_colors)]

            x, y, w, h = region['position']
            color = region['change_color']

            # Draw rectangles and text for individual changes
            cv2.rectangle(diff_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(diff_image, f"#{i + 1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Side-by-side image
            cv2.rectangle(side_by_side, (x, y), (x + w, y + h), color, 2)
            cv2.putText(side_by_side, f"#{i + 1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.rectangle(side_by_side, (x + image1.shape[1], y), (x + w + image1.shape[1], y + h), color, 2)
            cv2.putText(side_by_side, f"#{i + 1}", (x + image1.shape[1], y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Create transparent overlay
            overlay = highlight_overlay.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)

            # Apply transparency
            alpha = 0.3
            highlight_overlay = cv2.addWeighted(overlay, alpha, highlight_overlay, 1 - alpha, 0)

            # Add outline to highlighted area
            cv2.rectangle(highlight_overlay, (x, y), (x + w, y + h), color, 2)
            cv2.putText(highlight_overlay, f"#{i + 1}", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            difference_regions.append(region)

    total_pixels = image1.shape[0] * image1.shape[1]
    diff_pixels = np.sum(mask > 0)
    diff_percentage = (diff_pixels / total_pixels) * 100

    rgb_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    rgb_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    rgb_diff_image = cv2.cvtColor(diff_image, cv2.COLOR_BGR2RGB)
    rgb_composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    rgb_abs_diff = cv2.cvtColor(abs_diff, cv2.COLOR_BGR2RGB)
    rgb_highlight_overlay = cv2.cvtColor(highlight_overlay, cv2.COLOR_BGR2RGB)
    rgb_side_by_side = cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB)

    return {
        'image1': rgb_image1,
        'image2': rgb_image2,
        'diff_image': rgb_diff_image,
        'highlight_overlay': rgb_highlight_overlay,
        'side_by_side': rgb_side_by_side,
        'composite': rgb_composite,
        'abs_diff': rgb_abs_diff,
        'ssim_score': score,
        'diff_pixels': diff_pixels,
        'diff_percentage': diff_percentage,
        'difference_regions': difference_regions,
        'labels': labels,
        'has_text_detection': has_text_detection,
        'threshold': threshold,
        'grouping_enabled': grouping_enabled,
        'group_distance': group_distance
    }


class ZoomableChanges:
    """
    Class for interactively viewing and exploring image changes.
    """
    def __init__(self, results):
        self.results = results
        self.current_region_index = None
        self.fig = None
        self.base_axis = None
        self.zoom_axis = None

    def show_region(self, region_index):
        """
        Display a specific region of change with zoomed detail view.

        Parameters:
        - region_index: Index of the region to display
        """
        region = self.results['difference_regions'][region_index]

        if self.fig is None:
            self.fig, (self.base_axis, self.zoom_axis) = plt.subplots(1, 2, figsize=(18, 8))
            self.fig.suptitle(f"Detailed Change Analysis: Region #{region['id']}", fontsize=14)
        else:
            self.base_axis.clear()
            self.zoom_axis.clear()
            self.fig.suptitle(f"Detailed Change Analysis: Region #{region['id']}", fontsize=14)

        # Show the full image with current region highlighted
        self.base_axis.imshow(self.results['image1'])
        x, y, w, h = region['position']

        # Normalize color values from 0-255 to 0-1 range for matplotlib
        if isinstance(region['change_color'], (list, tuple)) and len(region['change_color']) >= 3:
            r, g, b = region['change_color'][:3]
            # Check if values appear to be in 0-255 range
            if max(r, g, b) > 1:
                edge_color = (r / 255.0, g / 255.0, b / 255.0)
            else:
                edge_color = region['change_color']
        else:
            # Fallback color if there's an issue with the color format
            edge_color = (1.0, 0.0, 0.0)  # Red

        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=edge_color, facecolor='none')
        self.base_axis.add_patch(rect)
        self.base_axis.set_title("Location in Original Image")
        self.base_axis.axis('off')

        # Show zoomed comparison
        zoom_fig = np.hstack((region['zoomed_region1'], region['zoomed_region2']))
        self.zoom_axis.imshow(zoom_fig)
        self.zoom_axis.set_title(f"Zoomed Comparison (Left: Original, Right: Modified)")
        self.zoom_axis.axis('off')

        # Add text details below
        details = f"Change Type: {region['change_type']}\n"
        details += f"Description: {region['detailed_change_description']}\n"

        # For grouped changes, add additional information
        if 'subregions' in region:
            details += f"\nThis group contains {region['contains']} related changes\n"

        if region['change_type'] == "Text Change":
            details += f"Old Text: '{region.get('old_text', '')}\n"
            details += f"New Text: '{region.get('new_text', '')}\n"
        elif region['change_type'] == "Color Change":
            details += f"Old Color: {region.get('old_color', '')}\n"
            details += f"New Color: {region.get('new_color', '')}\n"

        plt.figtext(0.5, 0.1, details, ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        plt.show()


def display_results(results):
    """
    Display the comparison results in a series of matplotlib figures.

    Parameters:
    - results: Dictionary containing comparison results from compare_images

    Returns:
    - ZoomableChanges object for interactive exploration
    """
    # Create a figure with more subplots for the enhanced visualizations
    plt.figure(figsize=(18, 14))

    plt.subplot(2, 3, 1)
    plt.title(results['labels'][0])
    plt.imshow(results['image1'])
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(results['labels'][1])
    plt.imshow(results['image2'])
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Side-by-Side Comparison with Changes')
    plt.imshow(results['side_by_side'])
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Highlighted Changes (with ID)')
    plt.imshow(results['highlight_overlay'])
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Old vs New (Red=Removed, Green=Added)')
    plt.imshow(results['composite'])
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Change Details')
    plt.axis('off')

    y_pos = 0.95
    plt.text(0.05, y_pos, f"Overall Similarity: {results['ssim_score']:.2f} (0-1 scale)", fontsize=10)
    y_pos -= 0.04
    plt.text(0.05, y_pos, f"Changed Area: {results['diff_percentage']:.1f}% of image", fontsize=10)
    y_pos -= 0.05

    if results.get('grouping_enabled', False):
        plt.text(0.05, y_pos, f"Change Grouping: Enabled (Distance: {results.get('group_distance', 50)}px)",
                 fontsize=10)
    else:
        plt.text(0.05, y_pos, f"Change Grouping: Disabled", fontsize=10)
    y_pos -= 0.05

    if not results.get('has_text_detection', False):
        plt.text(0.05, y_pos, "Note: Text detection not available.", fontsize=9, style='italic')
        plt.text(0.05, y_pos - 0.03, "Install pytesseract for text comparison.", fontsize=9, style='italic')
        y_pos -= 0.08

    if len(results['difference_regions']) > 0:
        plt.text(0.05, y_pos, f"Changes Found: {len(results['difference_regions'])}", fontsize=10, weight='bold')
        y_pos -= 0.04

        for region in sorted(results['difference_regions'], key=lambda x: x['id']):
            # For grouped changes, show the group info
            if 'subregions' in region:
                plt.text(0.05, y_pos, f"Group #{region['id']} - Contains {region['contains']} changes",
                         fontsize=9, weight='bold')
            else:
                plt.text(0.05, y_pos, f"Change #{region['id']} - {region['change_type']}",
                         fontsize=9, weight='bold')
            y_pos -= 0.03

            plt.text(0.1, y_pos, f"{region['detailed_change_description']}", fontsize=8)
            y_pos -= 0.03

            if region['change_type'] == "Text Change" and region.get('old_text') and region.get('new_text'):
                plt.text(0.1, y_pos, f"From: '{region['old_text']}'", fontsize=8)
                y_pos -= 0.025
                plt.text(0.1, y_pos, f"To: '{region['new_text']}'", fontsize=8)
                y_pos -= 0.025

            if y_pos < 0.1:
                plt.text(0.05, y_pos, "... more changes not shown (see full report) ...", fontsize=8, style='italic')
                break
    else:
        plt.text(0.05, y_pos, "No significant changes detected", fontsize=10)

    plt.tight_layout()
    plt.savefig('change_overview.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Now create an interactive display to show zoomed versions of each change
    zoomer = ZoomableChanges(results)

    print("\nDetailed Change Report:")
    print("======================")
    print(f"Found {len(results['difference_regions'])} changes between images")

    if results.get('grouping_enabled', False):
        total_subregions = sum([region.get('contains', 1) for region in results['difference_regions']])
        print(f"Grouped {total_subregions} individual changes into {len(results['difference_regions'])} groups")
        print(f"Grouping distance: {results.get('group_distance', 50)} pixels")

    for i, region in enumerate(results['difference_regions']):
        print(f"\nChange #{region['id']}:")

        if 'subregions' in region:
            print(f"  Group containing {region['contains']} related changes")

        print(f"  Type: {region['change_type']}")
        print(f"  Description: {region['detailed_change_description']}")
        print(f"  Position: x={region['position'][0]}, y={region['position'][1]}, " +
              f"width={region['position'][2]}, height={region['position'][3]}")
        print(f"  Affected Area: {region['area']} pixels")

        # Display the first region as an example
        if i == 0:
            zoomer.show_region(i)

    print("\nTo examine a specific change in detail, use:")
    print("  zoomer.show_region(change_index)")
    print("where change_index is 0 for the first change, 1 for the second, etc.")

    return zoomer