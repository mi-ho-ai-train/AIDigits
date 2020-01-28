package com.example.aidigits;

import android.app.Activity;
import android.graphics.Bitmap;
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

    private static class Probability {
        private String title;
        private float probability;

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
        // Loads bitmap into a TensorImage.
        float[] byteBuffer = convertBitmapToFloats(bitmap);
        int[] shape = new int[2];
        shape[0] = modelImageSizeX;
        shape[1] = modelImageSizeY;

        inputImageBuffer.load(byteBuffer, shape);

        // Creates processor for the TensorImage.
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());
        Log.i(LOG_TAG, String.format("load image x=%d, y=%d", bitmap.getWidth(), bitmap.getHeight()));
        Log.i(LOG_TAG, String.format("convert image to ImageX=%d, ImageY=%d", modelImageSizeX, modelImageSizeY));

        int numRoration = sensorOrientation / 90;
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                        .add(new ResizeOp(modelImageSizeX, modelImageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                        .add(new Rot90Op(numRoration))
                        .add(createPreprocessNormalizer())
                        .build();
        TensorImage tensorImage = imageProcessor.process(inputImageBuffer);
        int[] tensorShape = tensorImage.getTensorBuffer().getShape();
        Log.i(LOG_TAG, String.format("tensorImage X=%d, Y=%d", tensorShape[0], tensorShape[1]));
        return tensorImage;
    }

    private float[] convertBitmapToFloats(Bitmap bitmap) {
        int maxX = bitmap.getWidth();
        int maxY = bitmap.getHeight();
        Log.i(LOG_TAG, String.format("convertBitmapToFloats X=%d, Y=%d", maxX, maxY));
        float[] result = new float[maxX * maxY];
        for (int x = 0; x < maxX; x++) {
            for (int y = 0; y < maxY; y++) {
                int color = bitmap.getPixel(x, y);
                int r = color >> 16 & 0xFF;
                int g = color >> 8 & 0xFF;
                int b = color & 0xFF;

                float greyColor = (r + g + b) / 3.0f / 255.0f;
                result[x + y * maxX] = greyColor;
            }
        }
        Log.i(LOG_TAG, String.format("convertedBitmapToFloats size=%d", result.length));
        return result;
    }

    private static Probability calculateTopProbability(Map<String, Float> labelProb) {
        String maxLabel = "";
        float maxProbability = 0;
        for (Map.Entry<String, Float> entry : labelProb.entrySet()) {
            if (entry.getValue().floatValue() > maxProbability) {
                maxProbability = entry.getValue().floatValue();
                maxLabel = entry.getKey();
            }
        }
        return new Probability(maxLabel, maxProbability);
    }


    public Probability recognizeImage(final Bitmap bitmap, int sensorOrientation) {
        inputImageBuffer = loadImage(bitmap, sensorOrientation);

        tfLite.run(inputImageBuffer.getBuffer(), outputProbabilityBuffer.getBuffer().rewind());

        // Gets the map of label and probability.
        Map<String, Float> labeledProbability =
                new TensorLabel(LABELS, probabilityProcessor.process(outputProbabilityBuffer))
                        .getMapWithFloatValue();

        return calculateTopProbability(labeledProbability);
    }

    public void close() {
        if (tfLite != null) {
            tfLite.close();
            tfLite = null;
        }
        tfLiteModel = null;
    }
}
