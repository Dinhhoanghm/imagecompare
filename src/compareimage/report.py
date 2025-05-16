"""
Report generation functionality for image comparison results.

This module contains functions for generating HTML, text, and feature-based
reports from image comparison results.
"""

import os
import matplotlib.pyplot as plt
import cv2


def generate_report(results, output_file="change_report.html"):
    """
    Generate a detailed HTML report with interactive elements.

    Parameters:
    - results: Results dictionary from compare_images
    - output_file: Path to save the HTML report
    """
    # Create directory for report images if it doesn't exist
    os.makedirs('report_images', exist_ok=True)

    # Save all the images to files first
    plt.imsave('report_images/image1.png', results['image1'])
    plt.imsave('report_images/image2.png', results['image2'])
    plt.imsave('report_images/highlighted.png', results['highlight_overlay'])
    plt.imsave('report_images/side_by_side.png', results['side_by_side'])
    plt.imsave('report_images/composite.png', results['composite'])

    # Save zoom images for each change
    for region in results['difference_regions']:
        region_id = region['id']
        plt.imsave(f'report_images/region_{region_id}_old.png', region['zoomed_region1'])
        plt.imsave(f'report_images/region_{region_id}_new.png', region['zoomed_region2'])

    # Create HTML content
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Change Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            .image-container { display: flex; margin-bottom: 20px; }
            .image-container img { max-width: 45%; margin-right: 10px; }
            .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
            .change-item { background-color: #fff; border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
            .change-images { display: flex; margin-top: 10px; }
            .change-images img { max-width: 45%; margin-right: 10px; border: 1px solid #ddd; }
            .highlight { background-color: #ffffcc; }
            .tabs { display: flex; margin-bottom: 10px; }
            .tab { padding: 10px 15px; cursor: pointer; background-color: #ddd; margin-right: 5px; }
            .tab.active { background-color: #f5f5f5; font-weight: bold; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .image-map-container { position: relative; display: inline-block; }
            .map-highlight { position: absolute; border: 2px solid; cursor: pointer; }
            .tooltip { display: none; position: absolute; background: white; border: 1px solid #ddd; padding: 5px; z-index: 100; }
            .group-info { background-color: #e8f4f8; border-left: 3px solid #4a90e2; padding: 10px; margin: 5px 0; }
        </style>
        <script>
            function showTab(tabId) {
                // Hide all tab contents
                var tabContents = document.getElementsByClassName('tab-content');
                for (var i = 0; i < tabContents.length; i++) {
                    tabContents[i].classList.remove('active');
                }

                // Deactivate all tabs
                var tabs = document.getElementsByClassName('tab');
                for (var i = 0; i < tabs.length; i++) {
                    tabs[i].classList.remove('active');
                }

                // Show the selected tab and content
                document.getElementById(tabId).classList.add('active');
                document.getElementById(tabId + '-tab').classList.add('active');
            }

            function showTooltip(id) {
                document.getElementById('tooltip-' + id).style.display = 'block';
            }

            function hideTooltip(id) {
                document.getElementById('tooltip-' + id).style.display = 'none';
            }

            function scrollToChange(id) {
                document.getElementById('change-' + id).scrollIntoView({ behavior: 'smooth' });
                // Highlight the element briefly
                document.getElementById('change-' + id).classList.add('highlight');
                setTimeout(function() {
                    document.getElementById('change-' + id).classList.remove('highlight');
                }, 1500);
            }
        </script>
    </head>
    <body>
        <h1>Image Change Report</h1>

        <div class="tabs">
            <div id="overview-tab" class="tab active" onclick="showTab('overview')">Overview</div>
            <div id="detailed-changes-tab" class="tab" onclick="showTab('detailed-changes')">Detailed Changes</div>
            <div id="interactive-tab" class="tab" onclick="showTab('interactive')">Interactive View</div>
            <div id="technical-tab" class="tab" onclick="showTab('technical')">Technical Details</div>
        </div>

        <div id="overview" class="tab-content active">
            <div class="summary">
                <h2>Image Comparison Summary</h2>
                <p>Comparing: <b>{results['labels'][0]}</b> with <b>{results['labels'][1]}</b></p>
                <p>Overall Similarity: <b>{results['ssim_score']:.2f}</b> (0-1 scale, higher means more similar)</p>
                <p>Changed Area: <b>{results['diff_percentage']:.1f}%</b> of the image</p>
                <p>Number of distinct changes: <b>{len(results['difference_regions'])}</b></p>"""

    # Add grouping info if enabled
    if results.get('grouping_enabled', False):
        total_subregions = sum([region.get('contains', 1) for region in results['difference_regions']])
        html += f"""
                <p>Change grouping: <b>Enabled</b> (grouped {total_subregions} individual changes into {len(results['difference_regions'])} groups)</p>
                <p>Grouping distance: <b>{results.get('group_distance', 50)} pixels</b></p>"""

    html += """
            </div>

            <h3>Overview Images</h3>
            <div class="image-container">
                <img src="report_images/image1.png" alt="Original Image">
                <img src="report_images/image2.png" alt="Modified Image">
            </div>

            <h3>Changes Highlighted</h3>
            <div class="image-map-container">
                <img src="report_images/highlighted.png" alt="Changes Highlighted">
                """

    # Add image map overlay
    for region in results['difference_regions']:
        region_id = region['id']
        x, y, w, h = region['position']

        # Normalize RGB values from 0-255 to 0-1 range for CSS
        if isinstance(region['change_color'], (list, tuple)) and len(region['change_color']) >= 3:
            r, g, b = region['change_color'][:3]
            # Check if values appear to be in 0-255 range
            if max(r, g, b) > 1:
                r, g, b = r / 255.0, g / 255.0, b / 255.0
            color = f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 1.0)"
        else:
            # Fallback if change_color is not in expected format
            color = "rgba(255, 0, 0, 1.0)"  # Default to red

        # Different tooltip based on whether it's a group or individual change
        if 'subregions' in region:
            tooltip_text = f"Group #{region_id}: Contains {region['contains']} changes<br>{region['change_description']}"
        else:
            tooltip_text = f"Change #{region_id}: {region['change_type']}<br>{region['change_description']}"

        html += f"""
                <div class="map-highlight" style="left: {x}px; top: {y}px; width: {w}px; height: {h}px; border-color: {color};"
                    onmouseover="showTooltip({region_id})" onmouseout="hideTooltip({region_id})" onclick="scrollToChange({region_id})">
                </div>
                <div id="tooltip-{region_id}" class="tooltip" style="left: {x + w + 5}px; top: {y}px;">
                    {tooltip_text}
                </div>
                """

    # Continue building HTML with remaining sections
    html += """
            </div>

            <h3>Side-by-Side Comparison</h3>
            <div class="image-container">
                <img src="report_images/side_by_side.png" alt="Side-by-Side Comparison">
            </div>

            <h3>Removed vs Added Content</h3>
            <div class="image-container">
                <img src="report_images/composite.png" alt="Removed vs Added Content">
            </div>
        </div>

        <div id="detailed-changes" class="tab-content">
            <h2>Detailed Changes</h2>
            """

    # Add each change detail
    for region in results['difference_regions']:
        region_id = region['id']
        x, y, w, h = region['position']

        # Different header based on whether it's a group
        if 'subregions' in region:
            change_header = f"<h3>Group #{region_id}: Contains {region['contains']} related changes</h3>"
        else:
            change_header = f"<h3>Change #{region_id}: {region['change_type']}</h3>"

        html += f"""
            <div id="change-{region_id}" class="change-item">
                {change_header}
                <p><b>Description:</b> {region['detailed_change_description']}</p>
                <p><b>Location:</b> x={x}, y={y}, width={w}, height={h}</p>
                """

        # For groups, include subregion details
        if 'subregions' in region:
            html += f"""
                <div class="group-info">
                    <p><b>This group contains {region['contains']} related changes:</b></p>
                    <ul>
                """

            # Add list of subregions
            for i, subregion in enumerate(region.get('subregions', [])):
                html += f"""
                        <li>Change {i + 1}: {subregion['change_type']} - {subregion['change_description']}</li>
                """

            html += """
                    </ul>
                </div>
                """

        # Standard details
        html += f"""
                <p><b>Change Details:</b></p>
                <ul>
                """

        if region['change_type'] == "Text Change" and region.get('old_text') and region.get('new_text'):
            html += f"""
                    <li>Old Text: '{region['old_text']}'</li>
                    <li>New Text: '{region['new_text']}'</li>
                """
        elif region['change_type'] == "Color Change" and 'old_color' in region and 'new_color' in region:
            html += f"""
                    <li>Old Color: {region['old_color']}</li>
                    <li>New Color: {region['new_color']}</li>
                """

        html += f"""
                </ul>

                <div class="change-images">
                    <img src="report_images/region_{region_id}_old.png" alt="Original Region">
                    <img src="report_images/region_{region_id}_new.png" alt="Modified Region">
                </div>
            </div>
            """

    # Add interactive view tab
    html += """
        </div>

        <div id="interactive" class="tab-content">
            <h2>Interactive View</h2>

            <div class="image-map-container">
                <img src="report_images/highlighted.png" alt="Interactive View" usemap="#image-map">
                <map name="image-map">
                """

    # Add clickable areas for each change
    for region in results['difference_regions']:
        region_id = region['id']
        x, y, w, h = region['position']
        html += f"""
                    <area shape="rect" coords="{x},{y},{x + w},{y + h}" href="#change-{region_id}" 
                          alt="Change #{region_id}" onclick="scrollToChange({region_id}); return false;">
                """

    html += """
                </map>
            </div>

            <h3>Change Index</h3>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Type</th>
                    <th>Description</th>
                    <th>Location</th>
                    <th>Action</th>
                </tr>
                """

    # Add table rows for each change
    for region in results['difference_regions']:
        region_id = region['id']
        x, y, w, h = region['position']

        # Adjust display based on whether it's a group
        if 'subregions' in region:
            change_type = f"Group ({region['change_type']})"
            description = f"Contains {region['contains']} related changes"
        else:
            change_type = region['change_type']
            description = region['change_description']

        html += f"""
                <tr>
                    <td>{region_id}</td>
                    <td>{change_type}</td>
                    <td>{description}</td>
                    <td>x={x}, y={y}</td>
                    <td><a href="#" onclick="scrollToChange({region_id}); return false;">View</a></td>
                </tr>
                """

    # Add technical details tab
    html += """
            </table>
        </div>

        <div id="technical" class="tab-content">
            <h2>Technical Details</h2>

            <h3>Analysis Parameters</h3>
            <ul>
                <li>SSIM (Structural Similarity Index): {:.4f}</li>
                <li>Difference threshold: {}</li>
                <li>Total pixels in image: {:,}</li>
                <li>Changed pixels: {:,}</li>
                <li>Text detection available: {}</li>
            </ul>

            <h3>Change Type Distribution</h3>
            """.format(
        results['ssim_score'],
        results['threshold'],
        results['image1'].shape[0] * results['image1'].shape[1],
        results['diff_pixels'],
        "Yes" if results['has_text_detection'] else "No"
    )

    # Count types of changes
    change_types = {}
    for region in results['difference_regions']:
        change_type = region['change_type']
        change_types[change_type] = change_types.get(change_type, 0) + 1

    html += """
            <table>
                <tr>
                    <th>Change Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                """

    total_changes = len(results['difference_regions'])
    for change_type, count in change_types.items():
        percentage = (count / total_changes) * 100 if total_changes > 0 else 0
        html += f"""
                <tr>
                    <td>{change_type}</td>
                    <td>{count}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
                """

    html += """
            </table>

            <h3>Detailed Analysis Methodology</h3>
            <p>This report was generated using the following techniques:</p>
            <ul>
                <li>Structural Similarity Index (SSIM) for overall image comparison</li>
                <li>Pixel-by-pixel difference detection with threshold {}</li>
                <li>Contour analysis to identify distinct change regions</li>
                <li>Color analysis using RGB distance calculation</li>
                <li>Edge detection to identify structural changes</li>
                <li>Brightness and contrast differential analysis</li>
            </ul>
            """.format(results['threshold'])

    # Add info about grouping if enabled
    if results.get('grouping_enabled', False):
        html += f"""
            <h3>Change Grouping Methodology</h3>
            <p>Changes were grouped using the following approach:</p>
            <ul>
                <li>DBSCAN clustering algorithm was used to group nearby changes</li>
                <li>Maximum distance between changes in the same group: {results.get('group_distance', 50)} pixels</li>
                <li>All changes within this distance of each other are grouped together</li>
                <li>Each group is assigned a common color and ID for easier identification</li>
            </ul>
            """

    html += """
        </div>
    </body>
    </html>
    """

    # Write the HTML to file
    with open(output_file, 'w') as f:
        f.write(html)

    print(f"\nDetailed HTML report generated: {output_file}")
    print("Open this file in a web browser to view the interactive report")

    # Also generate a text version for compatibility
    generate_text_report(results, output_file.replace('.html', '.txt'))


def generate_text_report(results, output_file="change_report.txt"):
    """
    Generate a text-based report for compatibility.

    Parameters:
    - results: Results dictionary from compare_images
    - output_file: Path to save the text report
    """
    with open(output_file, 'w') as f:
        f.write("IMAGE CHANGE REPORT\n")
        f.write("==================\n\n")

        f.write(f"Comparing:\n")
        f.write(f"  - {results['labels'][0]}\n")
        f.write(f"  - {results['labels'][1]}\n\n")

        f.write(f"Overall Similarity: {results['ssim_score']:.2f} (0-1 scale, higher means more similar)\n")
        f.write(f"Changed Area: {results['diff_percentage']:.1f}% of the image\n\n")

        # Add grouping info if enabled
        if results.get('grouping_enabled', False):
            total_subregions = sum([region.get('contains', 1) for region in results['difference_regions']])
            f.write(
                f"Change grouping: Enabled (grouped {total_subregions} individual changes into {len(results['difference_regions'])} groups)\n")
            f.write(f"Grouping distance: {results.get('group_distance', 50)} pixels\n\n")

        if len(results['difference_regions']) > 0:
            f.write(f"CONTENT CHANGES DETECTED ({len(results['difference_regions'])} total)\n")
            f.write("------------------------\n\n")

            text_changes = []
            color_changes = []
            content_changes = []
            groups = []

            for region in results['difference_regions']:
                if 'subregions' in region:
                    groups.append(region)
                elif region['change_type'] == "Text Change":
                    text_changes.append(region)
                elif region['change_type'] == "Color Change":
                    color_changes.append(region)
                else:
                    content_changes.append(region)

            if groups:
                f.write(f"GROUPED CHANGES ({len(groups)} groups):\n")
                for i, group in enumerate(groups, 1):
                    f.write(f"  {i}. Group #{group['id']}: Contains {group['contains']} related changes\n")
                    f.write(f"     Description: {group['detailed_change_description']}\n")
                    f.write(
                        f"     Location: x={group['position'][0]}, y={group['position'][1]}, width={group['position'][2]}, height={group['position'][3]}\n")

                    # List subregions
                    f.write("     Contains:\n")
                    for j, subregion in enumerate(group.get('subregions', []), 1):
                        f.write(
                            f"       - Change {j}: {subregion['change_type']} - {subregion['change_description']}\n")

                f.write("\n")

            if text_changes:
                f.write(f"TEXT CHANGES ({len(text_changes)} found):\n")
                for i, change in enumerate(text_changes, 1):
                    f.write(f"  {i}. {change['change_description']}\n")
                    x, y, w, h = change['position']
                    f.write(f"     Location: x={x}, y={y}, width={w}, height={h}\n")
                    f.write(f"     Old Text: '{change['old_text']}'\n")
                    f.write(f"     New Text: '{change['new_text']}'\n")
                f.write("\n")

            if color_changes:
                f.write(f"COLOR CHANGES ({len(color_changes)} found):\n")
                for i, change in enumerate(color_changes, 1):
                    f.write(f"  {i}. {change['change_description']}\n")
                    x, y, w, h = change['position']
                    f.write(f"     Location: x={x}, y={y}, width={w}, height={h}\n")
                    f.write(f"     Old Color: {change['old_color']}\n")
                    f.write(f"     New Color: {change['new_color']}\n")
                f.write("\n")

            if content_changes:
                f.write(f"OTHER CONTENT CHANGES ({len(content_changes)} found):\n")
                for i, change in enumerate(content_changes, 1):
                    f.write(f"  {i}. {change['detailed_change_description']}\n")
                    x, y, w, h = change['position']
                    f.write(f"     Location: x={x}, y={y}, width={w}, height={h}\n")
                    f.write(f"     Change Type: {change.get('detailed_change_type', 'Content Change')}\n")
                f.write("\n")
        else:
            f.write("No significant content changes detected.\n")

    print(f"Text report saved to {output_file}")


def generate_feature_differences(results, output_file="feature_differences.txt"):
    """
    Generate a text file describing differences in a specific feature-oriented format.

    Parameters:
    - results: Results dictionary from compare_images
    - output_file: Path to save the feature differences report
    """
    with open(output_file, 'w') as f:
        f.write(f"FEATURE DIFFERENCES BETWEEN {results['labels'][0]} AND {results['labels'][1]}\n")
        f.write("===================================================================\n\n")

        if len(results['difference_regions']) == 0:
            f.write("No significant differences were detected between the images.\n")
            return

        # Process each difference region
        for region in results['difference_regions']:
            if 'subregions' in region:
                # This is a group
                f.write(f"DIFFERENCE GROUP {region['id']}:\n")
                f.write(
                    f"The image 1 is different from image 2 in the feature: Group of {region['contains']} related changes\n")

                # Description based on group type
                if region['change_type'] == "Text Change" and region.get('old_text') and region.get('new_text'):
                    f.write(f"  - Text has changed from '{region['old_text']}' to '{region['new_text']}'\n")
                else:
                    f.write(f"  - {region['detailed_change_description']}\n")

                # Location
                x, y, w, h = region['position']
                f.write(f"  - Location: Region at coordinates (x={x}, y={y}) with size {w}x{h} pixels\n")

                # List all subregions
                f.write("  - Contains the following specific differences:\n")
                for i, subregion in enumerate(region['subregions'], 1):
                    f.write(f"    {i}. {subregion['change_type']}: {subregion['change_description']}\n")

                # Visual nature of the change
                if any(sr['change_type'] == "Text Change" for sr in region['subregions']):
                    f.write("  - This group contains text modifications\n")
                if any(sr['change_type'] == "Color Change" for sr in region['subregions']):
                    f.write("  - This group contains color modifications\n")
                if any(sr['detailed_change_type'] == "Shape/Structure Change" for sr in region['subregions']):
                    f.write("  - This group contains structural modifications\n")

            else:
                # This is an individual change
                f.write(f"DIFFERENCE {region['id']}:\n")
                f.write(f"The image 1 is different from image 2 in the feature: {region['change_type']}\n")

                # Detail based on change type
                if region['change_type'] == "Text Change":
                    f.write(f"  - Text has changed from '{region['old_text']}' to '{region['new_text']}'\n")
                elif region['change_type'] == "Color Change":
                    f.write(f"  - Color has changed from {region['old_color']} to {region['new_color']}\n")
                else:
                    f.write(f"  - {region['detailed_change_description']}\n")

                # Location
                x, y, w, h = region['position']
                f.write(f"  - Location: Feature at coordinates (x={x}, y={y}) with size {w}x{h} pixels\n")

                # Technical details
                if 'detail_analysis' in region:
                    f.write("  - Technical details:\n")
                    f.write(
                        f"    * Percentage of region changed: {region['detail_analysis']['diff_percentage_in_region']:.1f}%\n")
                    f.write(
                        f"    * Edge/structure changes: {region['detail_analysis']['edge_change_percentage']:.1f}%\n")
                    f.write(f"    * Brightness difference: {region['detail_analysis']['brightness_change']:.1f}\n")
                    f.write(f"    * Contrast difference: {region['detail_analysis']['contrast_change']:.1f}\n")

            # Visual description of what happened
            if region.get('difference_type') == "Added":
                f.write("  - Content was added in image 2 that was not present in image 1\n")
            elif region.get('difference_type') == "Removed":
                f.write("  - Content was removed in image 2 that was present in image 1\n")

            f.write("\n")

        # Summary
        f.write("\nSUMMARY:\n")
        f.write(f"Total number of distinct differences: {len(results['difference_regions'])}\n")
        change_types = {}
        for region in results['difference_regions']:
            change_types[region['change_type']] = change_types.get(region['change_type'], 0) + 1

        for change_type, count in change_types.items():
            f.write(f"  - {change_type}: {count} instances\n")

        f.write(
            f"\nOverall image similarity: {results['ssim_score']:.2f} on a scale of 0 to 1 (higher means more similar)\n")
        f.write(f"Percentage of image area affected by changes: {results['diff_percentage']:.1f}%\n")

    print(f"Feature differences saved to {output_file}")