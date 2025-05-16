"""
Command-line interface for the ImageDiff tool.

This module provides the main entry point for the command-line tool.
"""

import argparse
import os
import sys
from compareimage.compare import compare_images, display_results
from compareimage.report import generate_report, generate_feature_differences


def main():
    """
    Main entry point for the imagediff command-line tool.
    """
    parser = argparse.ArgumentParser(
        description='Compare two images and identify content changes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('image1',nargs='?', default='/home/hoangdd/Downloads/Incidents_1.png')
    parser.add_argument('image2',nargs='?', default='/home/hoangdd/Downloads/Incidents.png')
    parser.add_argument('--old-label', default='Original Image', help='Label for the first image')
    parser.add_argument('--new-label', default='Modified Image', help='Label for the second image')
    parser.add_argument('--report',nargs='?', default='change_report.html', help='Output file for the HTML report')
    parser.add_argument('--save-images', action='store_true', help='Save output images')
    parser.add_argument('--output-dir', default='report_images', help='Directory to save output images')
    parser.add_argument('--threshold', type=int, default=30,
                        help='Threshold for detecting differences (0-255, higher = less sensitive)')
    parser.add_argument('--no-display', action='store_true', help='Do not display visual results')
    parser.add_argument('--no-group', action='store_true', help='Do not group nearby changes together')
    parser.add_argument('--group-distance', type=int, default=50,
                        help='Maximum distance between changes to be grouped (in pixels)')
    parser.add_argument('--feature-diff',nargs='?', default='feature_differences.txt',
                        help='Output file for feature differences in plain text format')
    parser.add_argument('--version', action='store_true', help='Display version information and exit')

    args = parser.parse_args()

    # Handle version display
    if args.version:
        from compareimage import __version__
        print(f"ImageDiff version {__version__}")
        return 0

    try:
        # Check if input files exist
        if not os.path.isfile(args.image1):
            print(f"Error: First image file not found: {args.image1}")
            return 1

        if not os.path.isfile(args.image2):
            print(f"Error: Second image file not found: {args.image2}")
            return 1

        print(f"Comparing images: {args.image1} and {args.image2}")
        print(f"Using threshold: {args.threshold}")

        if not args.no_group:
            print(f"Change grouping enabled with distance: {args.group_distance} pixels")
        else:
            print("Change grouping disabled")

        # Perform the comparison
        results = compare_images(
            args.image1,
            args.image2,
            (args.old_label, args.new_label),
            threshold=args.threshold,
            grouping_enabled=not args.no_group,
            group_distance=args.group_distance
        )

        if not args.no_display:
            zoomer = display_results(results)
            print("\nUse zoomer.show_region(index) to examine specific changes")

        # Generate reports
        generate_report(results, args.report)
        print(f"HTML report saved to {args.report}")

        generate_feature_differences(results, args.feature_diff)
        print(f"Feature differences saved to {args.feature_diff}")

        if args.save_images:
            import cv2
            os.makedirs(args.output_dir, exist_ok=True)

            cv2.imwrite(os.path.join(args.output_dir, 'changes_highlighted.jpg'),
                        cv2.cvtColor(results['diff_image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.output_dir, 'old_vs_new.jpg'),
                        cv2.cvtColor(results['composite'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(args.output_dir, 'side_by_side.jpg'),
                        cv2.cvtColor(results['side_by_side'], cv2.COLOR_RGB2BGR))

            print(f"\nOutput images saved to {args.output_dir}")

        # Print summary
        print(f"\nImage Comparison Summary:")
        print(f"Overall Similarity: {results['ssim_score']:.2f} (0-1 scale)")
        print(f"Changed Area: {results['diff_percentage']:.1f}% of the image")
        print(f"Number of distinct changes: {len(results['difference_regions'])}")

        if len(results['difference_regions']) > 0:
            print("\nChanges overview:")
            for region in sorted(results['difference_regions'], key=lambda x: x['id']):
                if 'subregions' in region:
                    print(f"Group #{region['id']} - Contains {region['contains']} related changes")
                else:
                    print(f"Change #{region['id']} - {region['change_type']}: {region['detailed_change_description']}")
                print(f"  Location: x={region['position'][0]}, y={region['position'][1]}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())