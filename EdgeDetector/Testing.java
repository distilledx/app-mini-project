import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;

public class Testing {
    public static void main(String[] args) {
        try {

            // Loading image path
            String path = args[0];

            // Getting image from path
            BufferedImage image = ImageIO.read(new File(path));

            // Converting image if incompatible type
            if (image.getType() != BufferedImage.TYPE_INT_ARGB) {

                BufferedImage convertedImage = new BufferedImage(
                        image.getWidth(),
                        image.getHeight(),
                        BufferedImage.TYPE_INT_ARGB);

                for (int x = 0; x < image.getWidth(); x++) {
                    for (int y = 0; y < image.getHeight(); y++) {
                        int pixel = image.getRGB(x, y);
                        convertedImage.setRGB(x, y, pixel);
                    }
                }

                image = convertedImage;
            }

            // Create an instance of the CannyEdgeDetector
            CannyEdgeDetector detector = new CannyEdgeDetector();

            // Set the source image
            detector.setSourceImage(image);

            // Adjust the parameters as desired (optional)
            detector.setLowThreshold(0.5f);
            detector.setHighThreshold(1.0f);
            detector.setGaussianKernelRadius(2.0f);
            detector.setGaussianKernelWidth(16);
            detector.setContrastNormalized(false); // You can set this to true if needed

            // Apply the edge detection
            detector.process();

            // Get the resulting edges image
            BufferedImage edgesImage = detector.getEdgesImage();

            // Save or display the edges image
            File outputImage = new File(path.substring(0, path.lastIndexOf('.')) + "_edge.png");
            ImageIO.write(edgesImage, "png", outputImage);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
