package com.example.aidigits;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final String LOG_TAG = "Digits.Main";
    private static final String MODEL_FILENAME = "digits.tflite";
    public static final String[] TEST_FILE_NAMES = {"Digit1b.jpg", "Digit7b.jpg", "Digit8b.jpg"};

    ImageView imageView;
    Bitmap imageBitmap;
    Button button;
    Random rand = new Random();

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            Bundle extras = data.getExtras();
            imageBitmap = (Bitmap) extras.get("data");
            imageView.setImageBitmap(imageBitmap);
            int widthInPixel = imageBitmap.getWidth();
            int heightInPixel = imageBitmap.getHeight();
            Toast.makeText(this, String.format("Width=%d, Height=%d", widthInPixel, heightInPixel), Toast.LENGTH_LONG).show();
        }
    }

    public void onClickTakePictureButton(View view) {
        Log.i(LOG_TAG, "Take Picture Button clicked");
        dispatchTakePictureIntent();
    }

    public void onClickTestButton(View view) {
        Log.i(LOG_TAG, "Test Button clicked");
        Bitmap testBitmap;
        try {
            InputStream ims = getAssets().open(TEST_FILE_NAMES[rand.nextInt(TEST_FILE_NAMES.length)]);
            Log.i(LOG_TAG, "Asset opened");
            testBitmap = BitmapFactory.decodeStream(ims);
        } catch (IOException e) {
            Log.e(LOG_TAG, Log.getStackTraceString(e));
            return;
        }
        Log.i(LOG_TAG, String.format("Analyze Test Image of size (%d,%d)", testBitmap.getWidth(), testBitmap.getHeight()));
        analyzeImage(testBitmap);
    }

    public void onClickAnalyzeButton(View view) {
        Log.i(LOG_TAG, "Analyze Button clicked");
        analyzeImage(imageBitmap);
    }

    private void analyzeImage(Bitmap rawImageBitmap) {
        Bitmap imageBitmap = preprocessImage(rawImageBitmap);
        try {
            Log.i(LOG_TAG, "Analyze Image");
            ImageClassifier classifier = new ImageClassifier(this, 1, loadModel());
            Log.i(LOG_TAG, "Classify Image");
            ImageClassifier.Probability probability = classifier.recognizeImage(imageBitmap, 0);
            TextView textView = findViewById(R.id.textView);
            textView.setText(String.format("Probability: %s", probability.toString()));
            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageBitmap(probability.getBitmap());
            imageView.invalidate();
        } catch (IOException e) {
            Log.e(LOG_TAG, Log.getStackTraceString(e));
        }
    }

    private Bitmap preprocessImage(Bitmap imageBitmap) {
        // @param contrast 0..10 1 is default
        // @param brightness -255..255 0 is default
        int contrast = 1;
        int brightness = 0;
        ColorMatrix cm = new ColorMatrix(new float[]
                {
                        contrast, 0, 0, 0, brightness,
                        0, contrast, 0, 0, brightness,
                        0, 0, contrast, 0, brightness,
                        0, 0, 0, 1, 0
                });
        Bitmap ret = Bitmap.createBitmap(imageBitmap.getWidth(), imageBitmap.getHeight(), imageBitmap.getConfig());
        Canvas canvas = new Canvas(ret);
        Paint paint = new Paint();
        paint.setColorFilter(new ColorMatrixColorFilter(cm));
        canvas.drawBitmap(imageBitmap, 0, 0, paint);
        // Black and white
        int width = imageBitmap.getWidth();
        int height = imageBitmap.getHeight();
        // create output bitmap
        Bitmap bmOut = Bitmap.createBitmap(width, height, imageBitmap.getConfig());
        // color information
        int A, RED, G, B;
        int pixel;
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                // get pixel color
                pixel = imageBitmap.getPixel(x, y);
                A = Color.alpha(pixel);
                RED = Color.red(pixel);
                G = Color.green(pixel);
                B = Color.blue(pixel);
                int gray = (int) (0.2989 * RED + 0.5870 * G + 0.1140 * B);
                gray = Math.abs(gray - 255);
                if (gray < 120) gray = 0;
                // set new pixel color to output bitmap
                bmOut.setPixel(x, y, Color.argb(A, gray, gray, gray));
            }
        }
        int newWidth = 28;
        int newHeight = 28;
        // calculate the scale - in this case = 0.4f
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // createa matrix for the manipulation
        Matrix matrix = new Matrix();
        // resize the bit map
        matrix.postScale(scaleWidth, scaleHeight);
        // recreate the new Bitmap
        return Bitmap.createBitmap(bmOut, 0, 0,
                width, height, matrix, true);
    }


    private ByteBuffer loadModel() {
        ByteBuffer tfLiteModel = ByteBuffer.allocate(0);
        AssetManager assetManager = getAssets();
        try {
            InputStream input = assetManager.open(MODEL_FILENAME);
            int size = input.available();
            byte[] bytes = new byte[size];
            int length = input.read(bytes);
            Log.i(LOG_TAG, String.format("Read Model file of %d bytes", length));
            tfLiteModel = ByteBuffer.allocateDirect(size);
            tfLiteModel.put(bytes);
            tfLiteModel.order(ByteOrder.nativeOrder());
        } catch (IOException e) {
            Log.e(LOG_TAG, Log.getStackTraceString(e));
        }
        return tfLiteModel;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        button = findViewById(R.id.buttonTakePicture);
    }
}
