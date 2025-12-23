#!/usr/bin/env python3
##"""
AUTOMATIC EXPOSURE ENHANCER - v1.0
Smart auto-exposure adjustment for product photos
Processes entire folders, preserves quality, overwrites with safety backup
"""

import sys
from pathlib import Path
from datetime import datetime
import shutil

try:
    from PIL import Image, ImageEnhance, ImageStat
except ImportError:
    print("ERROR: Pillow not installed!")
    print("Install with: pip install Pillow --break-system-packages")
    sys.exit(1)

# =====================================================
# SMART EXPOSURE ANALYZER
# =====================================================

class ExposureAnalyzer:
    """Analyze image brightness and determine enhancement needed"""

    @staticmethod
    def analyze_brightness(image):
        """
        Analyze image brightness (0-255 scale)
        Returns: average brightness value
        """
        # Convert to grayscale for brightness analysis
        grayscale = image.convert('L')
        stat = ImageStat.Stat(grayscale)

        # Get average brightness
        avg_brightness = stat.mean[0]

        return avg_brightness

    @staticmethod
    def calculate_enhancement_factor(brightness):
        """
        Smart calculation of how much to brighten
        Target: 140-160 brightness (good exposure for product photos)

        Returns: enhancement factor (1.0 = no change, >1.0 = brighter)
        """

        # Optimal brightness range for product photos
        TARGET_MIN = 140
        TARGET_MAX = 160
        TARGET_OPTIMAL = 150

        if brightness >= TARGET_MIN and brightness <= TARGET_MAX:
            # Already good exposure
            return 1.0

        elif brightness < TARGET_MIN:
            # Too dark - calculate enhancement needed
            # The darker it is, the more we enhance

            if brightness < 50:
                # Very dark - aggressive enhancement
                factor = 2.2
            elif brightness < 80:
                # Dark - strong enhancement
                factor = 1.8
            elif brightness < 110:
                # Somewhat dark - moderate enhancement
                factor = 1.5
            else:
                # Slightly dark - gentle enhancement
                factor = 1.2

            return factor

        else:
            # Too bright - slight reduction (rare for product photos)
            return 0.9

# =====================================================
# IMAGE ENHANCER
# =====================================================

class ImageEnhancer:
    """Enhance image exposure with smart adjustments"""

    def __init__(self):
        self.analyzer = ExposureAnalyzer()

    def enhance_image(self, image_path, overwrite=True, backup=True):
        """
        Enhance a single image

        Args:
            image_path: Path to image
            overwrite: If True, replace original
            backup: If True, create backup before overwriting

        Returns:
            dict with results
        """

        try:
            # Load image
            img = Image.open(image_path)
            original_mode = img.mode

            # Analyze brightness
            original_brightness = self.analyzer.analyze_brightness(img)

            # Calculate enhancement factor
            factor = self.analyzer.calculate_enhancement_factor(original_brightness)

            # Check if enhancement needed
            if factor == 1.0:
                return {
                    'status': 'skipped',
                    'reason': 'Already optimal exposure',
                    'original_brightness': round(original_brightness, 1),
                    'enhancement_factor': factor
                }

            # Apply enhancement
            enhancer = ImageEnhance.Brightness(img)
            enhanced_img = enhancer.enhance(factor)

            # Verify enhanced brightness
            enhanced_brightness = self.analyzer.analyze_brightness(enhanced_img)

            # Restore original mode (important for transparency)
            if enhanced_img.mode != original_mode:
                enhanced_img = enhanced_img.convert(original_mode)

            # Save image
            if overwrite:
                # Create backup if requested
                if backup:
                    backup_path = self._create_backup(image_path)
                else:
                    backup_path = None

                # Save enhanced image (overwrite original)
                enhanced_img.save(image_path, quality=95, optimize=True)

                return {
                    'status': 'enhanced',
                    'original_brightness': round(original_brightness, 1),
                    'enhanced_brightness': round(enhanced_brightness, 1),
                    'enhancement_factor': round(factor, 2),
                    'backup_path': backup_path
                }
            else:
                # Save as new file
                output_path = self._generate_output_path(image_path)
                enhanced_img.save(output_path, quality=95, optimize=True)

                return {
                    'status': 'enhanced',
                    'original_brightness': round(original_brightness, 1),
                    'enhanced_brightness': round(enhanced_brightness, 1),
                    'enhancement_factor': round(factor, 2),
                    'output_path': output_path
                }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

    def _create_backup(self, image_path):
        """Create backup of original image"""
        path = Path(image_path)
        backup_dir = path.parent / 'backups'
        backup_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = backup_dir / f"{path.stem}_backup_{timestamp}{path.suffix}"

        shutil.copy2(image_path, backup_path)
        return str(backup_path)

    def _generate_output_path(self, image_path):
        """Generate output path for new file"""
        path = Path(image_path)
        return str(path.parent / f"{path.stem}_ENHANCED{path.suffix}")

# =====================================================
# BATCH PROCESSOR
# =====================================================

class BatchProcessor:
    """Process entire folders of images"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self):
        self.enhancer = ImageEnhancer()

    def process_folder(self, folder_path, overwrite=True, backup=True, recursive=False):
        """
        Process all images in a folder

        Args:
            folder_path: Path to folder
            overwrite: If True, replace originals
            backup: If True, create backups
            recursive: If True, process subfolders

        Returns:
            dict with summary
        """

        folder = Path(folder_path)

        if not folder.exists():
            print(f"❌ ERROR: Folder not found: {folder_path}")
            return None

        # Find all images
        if recursive:
            image_files = []
            for ext in self.SUPPORTED_FORMATS:
                image_files.extend(folder.rglob(f"*{ext}"))
                image_files.extend(folder.rglob(f"*{ext.upper()}"))
        else:
            image_files = []
            for ext in self.SUPPORTED_FORMATS:
                image_files.extend(folder.glob(f"*{ext}"))
                image_files.extend(folder.glob(f"*{ext.upper()}"))

        # Remove duplicates and sort
        image_files = sorted(set(image_files))

        if not image_files:
            print(f"⚠️  No images found in: {folder_path}")
            return None

        print(f"\n📁 Found {len(image_files)} images")
        print(f"🔧 Processing with auto-exposure enhancement...")
        if overwrite:
            print(f"⚠️  OVERWRITE MODE: Original files will be replaced")
            if backup:
                print(f"✅ Safety backups will be created in 'backups' folder")
        print("")

        # Process images
        results = {
            'total': len(image_files),
            'enhanced': 0,
            'skipped': 0,
            'errors': 0,
            'details': []
        }

        for idx, img_path in enumerate(image_files, 1):
            # Progress indicator
            progress = f"[{idx}/{len(image_files)}]"
            print(f"{progress} Processing: {img_path.name}...", end=' ')

            # Enhance image
            result = self.enhancer.enhance_image(
                str(img_path),
                overwrite=overwrite,
                backup=backup
            )

            # Update results
            if result['status'] == 'enhanced':
                results['enhanced'] += 1
                brightness_change = result['enhanced_brightness'] - result['original_brightness']
                print(f"✅ Enhanced! ({result['original_brightness']:.0f} → {result['enhanced_brightness']:.0f}, +{brightness_change:.0f})")
            elif result['status'] == 'skipped':
                results['skipped'] += 1
                print(f"⏭️  Skipped (already optimal: {result['original_brightness']:.0f})")
            else:
                results['errors'] += 1
                print(f"❌ Error: {result.get('error', 'Unknown')}")

            results['details'].append({
                'file': img_path.name,
                'result': result
            })

        return results

# =====================================================
# MAIN EXECUTION
# =====================================================

def main():
    """Main execution"""

    print("\n" + "="*80)
    print("📸 AUTOMATIC EXPOSURE ENHANCER")
    print("="*80 + "\n")

    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 auto_enhance_exposure.py <folder_path> [options]")
        print("\nOptions:")
        print("  --no-backup      Skip creating backups (not recommended)")
        print("  --no-overwrite   Save as new files instead of overwriting")
        print("  --recursive      Process subfolders too")
        print("\nDefault behavior:")
        print("  ✅ Auto-adjusts exposure intelligently")
        print("  ✅ Overwrites original files")
        print("  ✅ Creates safety backups in 'backups' folder")
        print("  ✅ Skips already well-exposed images")
        print("\nExamples:")
        print("  python3 auto_enhance_exposure.py /path/to/images")
        print("  python3 auto_enhance_exposure.py /path/to/images --recursive")
        print("  python3 auto_enhance_exposure.py /path/to/images --no-backup")
        return 1

    folder_path = sys.argv[1]

    # Parse options
    backup = '--no-backup' not in sys.argv
    overwrite = '--no-overwrite' not in sys.argv
    recursive = '--recursive' in sys.argv

    # Process folder
    processor = BatchProcessor()
    results = processor.process_folder(
        folder_path,
        overwrite=overwrite,
        backup=backup,
        recursive=recursive
    )

    if results:
        # Print summary
        print("\n" + "="*80)
        print("📊 PROCESSING SUMMARY")
        print("="*80)
        print(f"Total Images:     {results['total']}")
        print(f"✅ Enhanced:      {results['enhanced']}")
        print(f"⏭️  Skipped:       {results['skipped']} (already optimal)")
        print(f"❌ Errors:        {results['errors']}")
        print("="*80)

        if results['enhanced'] > 0:
            print("\n✅ Exposure enhancement complete!")
            if backup and overwrite:
                print(f"💾 Original images backed up in 'backups' folder")

        print("")
        return 0

    return 1

if __name__ == '__main__':
    exit(main())
