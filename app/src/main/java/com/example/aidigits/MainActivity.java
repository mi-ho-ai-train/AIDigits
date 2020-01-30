package com.example.aidigits;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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

public class MainActivity extends AppCompatActivity {

    private static final int REQUEST_IMAGE_CAPTURE = 1;
    private static final String LOG_TAG = "Digits.Main";
    private static final String MODEL_FILENAME = "digits.tflite";
    public static final String TEST_FILE_NAME = "Test.JPG";

    ImageView imageView;
    Bitmap imageBitmap;
    Button button;

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
            InputStream ims = getAssets().open(TEST_FILE_NAME);
            Log.i(LOG_TAG, "Asset opened");
            testBitmap = BitmapFactory.decodeStream(ims);
        }
        catch(IOException e) {
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

    private void analyzeImage(Bitmap imageBitmap) {
        try {
            Log.i(LOG_TAG, "Analyze Image");
            ImageClassifier classifier = new ImageClassifier(this, 1, loadModel());
            Log.i(LOG_TAG, "Classify Image");
            ImageClassifier.Probability probability =  classifier.recognizeImage(imageBitmap, 0);
            TextView textView = findViewById(R.id.textView);
            textView.setText(String.format("Probability: %s", probability.toString()));
            ImageView imageView = findViewById(R.id.imageView);
            imageView.setImageBitmap(probability.getBitmap());
            imageView.invalidate();
        } catch (IOException e) {
            Log.e(LOG_TAG, Log.getStackTraceString(e));
        }
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
