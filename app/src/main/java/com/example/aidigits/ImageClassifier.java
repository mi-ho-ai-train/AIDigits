package com.example.aidigits;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class ImageClassifier {
    private static final String LOG_TAG = "Digits.Classifier";

    private TensorImage inputImageBuffer;

    private MappedByteBuffer tfLiteModel;
    private Interpreter tfLite;
    private final Interpreter.Options tfLiteOptions = new Interpreter.Options();

    private static final float IMAGE_MEAN = 127.5f;
    private static final float IMAGE_STD = 127.5f;
    private static final float PROBABILITY_MEAN = 0.0f;
    private static final float PROBABILITY_STD = 1.0f;

    private static final int NUM_BYTES_PER_CHANNEL = 3;

    private final int modelImageSizeX;
    private final int modelImageSizeY;

    private final TensorBuffer outputProbabilityBuffer;
    private final TensorProcessor probabilityProcessor;

    private List<String> LABELS = Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9");
    private int color;

    public static class Probability {
        private String title;
        private float probability;
        private Bitmap bitmap;

        public Probability(String title, float probability) {
            this.title = title;
            this.probability = probability;
        }

        public String getTitle() {
            return title;
        }

        public float getProbability() {
            return probability;
        }

        public Bitmap getBitmap() {
            return bitmap;
        }

        @Override
        public String toString() {
            return String.format("[%s:%f]", title, probability);
        }

        public void setBitmap(Bitmap bitmap) {
            this.bitmap = bitmap;
        }
    }


    protected ImageClassifier(Activity activity, int numThreads, ByteBuffer tfLiteModel) throws IOException {
        tfLiteOptions.setNumThreads(numThreads);
        tfLite = new Interpreter(tfLiteModel, tfLiteOptions);

        // Reads type and shape of input and output tensors, respectively.
        int imageTensorIndex = 0;
        int[] imageShape = tfLite.getInputTensor(imageTensorIndex).shape();
        modelImageSizeY = imageShape[1];
        modelImageSizeX = imageShape[2];
        Log.i(LOG_TAG, String.format("ImageX=%d, ImageY=%d", modelImageSizeX, modelImageSizeY));

        DataType imageDataType = tfLite.getInputTensor(imageTensorIndex).dataType();
        int probabilityTensorIndex = 0;
        int[] probabilityShape =
                tfLite.getOutputTensor(probabilityTensorIndex).shape(); // {1, NUM_CLASSES}
        DataType probabilityDataType = tfLite.getOutputTensor(probabilityTensorIndex).dataType();

        // Creates the input tensor.
        inputImageBuffer = new TensorImage(imageDataType);

        // Creates the output tensor and its processor.
        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);

        // Creates the post processor for the output probability.
        probabilityProcessor = new TensorProcessor.Builder().add(createPostprocessNormalizer()).build();
    }


    private TensorOperator createPreprocessNormalizer() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    private TensorOperator createPostprocessNormalizer() {
        return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
    }

    private TensorImage loadImage(final Bitmap bitmap, int sensorOrientation) {
        int[] shape = new int[2];
        shape[0] = modelImageSizeX;
        shape[1] = modelImageSizeY;

        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        Log.i(LOG_TAG, String.format("load image x=%d, y=%d", bitmap.getWidth(), bitmap.getHeight()));
        Log.i(LOG_TAG, String.format("convert image to ImageX=%d, ImageY=%d", modelImageSizeX, modelImageSizeY));

        int rotationInQuaters = sensorOrientation / 90;
        DataType dataType = DataType.FLOAT32;
        TensorImage sourceImage = new TensorImage(dataType);
        sourceImage.load(bitmap);
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(modelImageSizeX, modelImageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(rotationInQuaters))
                        .build();
        TensorImage tensorImage = imageProcessor.process(sourceImage);
        return tensorImage;
    }

    private FloatBuffer convertTensorImageToFloatBuffer(TensorImage image) {
        int[] shape = image.getTensorBuffer().getShape();
        Log.i(LOG_TAG, String.format("convertBitmapToFloats shape 0=%d, 1=%d, 2=%d", shape[0], shape[1], shape[2]));
        int maxX = shape[0];
        int maxY = shape[1];
        float[] buffer = image.getTensorBuffer().getFloatArray();
        Log.i(LOG_TAG, String.format("convertBitmapToFloats X=%d, Y=%d, size buffer = %d", maxX, maxY, buffer.length));

        float[] result = new float[maxX * maxY];
        for (int x = 0; x < maxX; x++) {
            for (int y = 0; y < maxY; y++) {
                int i = (x + y * maxX) * 3;
                float r = buffer[i];
                float g = buffer[i + 1];
                float b = buffer[i + 2];

                float greyColor = (r + g + b) / 3.0f / 255.0f;
                result[x + y * maxX] = greyColor;
            }
        }
        Log.i(LOG_TAG, String.format("convertedBitmapToFloats size=%d", result.length));
        FloatBuffer floatBuffer = FloatBuffer.wrap(result);
        return floatBuffer;
    }

    private Bitmap convertFloatBufferToBitmap(FloatBuffer floatBuffer) {
        Bitmap bitmap = Bitmap.createBitmap(modelImageSizeX, modelImageSizeY, Bitmap.Config.ARGB_8888);
        for (int x = 0; x < modelImageSizeX; x++) {
            for (int y = 0; y < modelImageSizeY; y++) {
                int c = (int) (256 * floatBuffer.get(x + y * modelImageSizeX));
                int color = Color.rgb(c, c, c);
                bitmap.setPixel(x, y, color);
            }
        }
        return bitmap;
    }

    private static Probability calculateTopProbability(Map<String, Float> labelProb) {
        String maxLabel = "";
        float maxProbability = 0;
        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            Log.i(LOG_TAG, String.format("Probability key=%s, entry=%f", entry.getKey(), entry.getValue()));
            if (entry.getValue().floatValue() > maxProbability) {
                maxProbability = entry.getValue().floatValue();
                maxLabel = entry.getKey();
            }
        }
        return new Probability(maxLabel, maxProbability);
    }


    public Probability recognizeImage(final Bitmap bitmap, int sensorOrientation) {
        inputImageBuffer = loadImage(bitmap, sensorOrientation);
        FloatBuffer floatBuffer = convertTensorImageToFloatBuffer(inputImageBuffer);
        tfLite.run(floatBuffer, outputProbabilityBuffer.getBuffer().rewind());

        // Gets the map of label and probability.
        Map<String, Float> labeledProbability =
                new TensorLabel(LABELS, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();

        Probability probability = calculateTopProbability(labeledProbability);
        probability.setBitmap(convertFloatBufferToBitmap(floatBuffer));
        return probability;
    }

    public void close() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = null;
        }
        tfLiteModel = null;
    }
}
