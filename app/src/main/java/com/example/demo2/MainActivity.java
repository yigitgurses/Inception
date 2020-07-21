package com.example.demo2;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.core.content.FileProvider;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.text.SimpleDateFormat;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

//important
import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {

    // Constants
    static final int REQUEST_PHOTO_CAPTURE = 1;
    static final int REQUEST_GALLERY_LOAD = 2;
    static final int REQUEST_PERMISSION = 3;
    static final int NO_OF_RESULTS = 3;

    // input image dimensions for the Inception Model
    private int INPUT_X = 224;
    private int INPUT_Y = 224;
    private int INPUT_Z = 3; // each pixel holds rgb values

    // options for model interpreter
    private final Interpreter.Options tfliteOptions = new Interpreter.Options();
    // tflite graph
    private Interpreter tflite;
    // holds all the possible labels for model
    private List<String> labelList;
    // holds the selected image data as bytes
    private ByteBuffer imgData = null;
    // holds the probabilities of each label for quantized graphs
    private byte[][] probArray = null;
    // array that holds the labels with the highest probabilities
    private String[] topLabels = null;
    // array that holds the highest probabilities
    private String[] topConfidence = null;

    String currentPhotoPath;

    //Activity Elements
    ImageView imageView;
    Button btnTakePicture, btnLoadFromGallery;
    TextView txtPred1, txtPred2, txtPred3;

    private static final String TAG = "MainActivity";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //initilize graph and labels
        try{
            tflite = new Interpreter(loadModelFile(), tfliteOptions);
            labelList = loadLabelList();
        } catch (Exception ex){
            ex.printStackTrace();
        }

        // initialize byte buffer to send image data to model
        imgData = ByteBuffer.allocateDirect( INPUT_X * INPUT_Y * INPUT_Z );
        imgData.order(ByteOrder.nativeOrder());

        // initialize probabilities array
        probArray = new byte[1][labelList.size()];

        // initialize arrays to hold top labels and top confidences
        topLabels = new String[NO_OF_RESULTS];
        topConfidence = new String[NO_OF_RESULTS];

        // initialize activity elements
        imageView = findViewById(R.id.imageView);
        btnTakePicture = findViewById(R.id.btnTakePicture);
        btnLoadFromGallery = findViewById(R.id.btnLoadFromGallery);
        txtPred1 = findViewById(R.id.txtPred1);
        txtPred2 = findViewById(R.id.txtPred2);
        txtPred3 = findViewById(R.id.txtPred3);

        //set on click listeners for the buttons
        btnTakePicture.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dispatchTakePictureIntent();
            }
        });
        btnLoadFromGallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                dispatchLoadFromGalleryIntent();
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        // handling image capture, putting the captured image to imageView
        if (requestCode == REQUEST_PHOTO_CAPTURE && resultCode == RESULT_OK) {
            Bitmap imageBitmap = getBitmapFromPath(currentPhotoPath);
            imageView.setImageBitmap(imageBitmap);
            classify();
        }

        // handling loading from gallery, putting the loaded image to imageView
        if (requestCode == REQUEST_GALLERY_LOAD &&  resultCode == RESULT_OK) {
            try {
                Uri imageUri = data.getData();
                InputStream imageStream = getContentResolver().openInputStream(imageUri);
                Bitmap selectedImage = BitmapFactory.decodeStream(imageStream);
                imageView.setImageBitmap(selectedImage);
                classify();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        //TODO
    }

    //helper methods
    private void classify() {
        // get the bitmap in the image view
        Bitmap bitmap = ( (BitmapDrawable) imageView.getDrawable() ).getBitmap();
        // rescale it to proper input sizes
        bitmap = getResizedBitmap(bitmap, INPUT_X, INPUT_Y);
        convertBitmapToByteBuffer(bitmap);
        // run the model to get predictions
        tflite.run(imgData, probArray);
        // prints the best predictions on the textViews
        printTopKResults(NO_OF_RESULTS);
    }

    private void dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
            File photoFile = null;
            try {
                photoFile = createImageFile();
            } catch (IOException ex) {
                // Error occurred while creating the File
                Log.d(TAG, "dispatchTakePictureIntent: Could not create File");
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                Uri photoURI = FileProvider.getUriForFile(this,
                        "com.example.demo2.fileprovider",
                        photoFile);
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                startActivityForResult(takePictureIntent, REQUEST_PHOTO_CAPTURE);
            }
        }
    }

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File imageFile = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        currentPhotoPath = imageFile.getAbsolutePath();
        return imageFile;
    }

    private void dispatchLoadFromGalleryIntent() {
        Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
        photoPickerIntent.setType("image/*");
        startActivityForResult(photoPickerIntent, REQUEST_GALLERY_LOAD);
    }

    public Bitmap getBitmapFromPath(String path) {
        Bitmap bitmap = null;
        try {
            File f = new File(path);
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            bitmap = BitmapFactory.decodeStream(new FileInputStream(f), null, options);
        } catch (Exception e) {
            e.printStackTrace();
        }
        return bitmap;
    }

    // converts bitmap to byte array which is passed to the tflite graph
    private void convertBitmapToByteBuffer(Bitmap bitmap) {
        int noOfPixels = bitmap.getWidth() * bitmap.getHeight();
        int[] pixelValues = new int[noOfPixels];;
        bitmap.getPixels(pixelValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        // loop through all pixels
        imgData.rewind();
        for ( int i = 0; i < noOfPixels; i++ ) {
            final int val = pixelValues[i++];
            // get rgb values from intValues where each int holds the rgb values for a pixel.
            imgData.put((byte) (val >> 16) );
            imgData.put((byte) (val >> 8) );
            imgData.put((byte) (val));
        }
    }

    // loads tflite graph from file
    private MappedByteBuffer loadModelFile() throws IOException {
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model_quant.tflite");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // loads the labels from the label txt file in assets into a string array
    private List<String> loadLabelList() throws IOException {
        List<String> labelList = new ArrayList<String>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(this.getAssets().open("labels.txt")));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    // resizes bitmap to given dimensions
    public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
        int width = bm.getWidth();
        int height = bm.getHeight();
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        Bitmap resizedBitmap = Bitmap.createBitmap(
                bm, 0, 0, width, height, matrix, false);
        return resizedBitmap;
    }

    // print the top labels and respective confidences
    private void printTopKResults( int k ) {
        // priority queue that will hold the top results from the CNN
        PriorityQueue<Map.Entry<String, Float>> sortedLabels = new PriorityQueue<>(
                        NO_OF_RESULTS,
                        new Comparator<Map.Entry<String, Float>>() {
                            @Override
                            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
                                return (o1.getValue()).compareTo(o2.getValue());
                            }
                        });

        // add all results to priority queue
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(
                    new AbstractMap.SimpleEntry<>(labelList.get(i), (probArray[0][i] & 0xff) / 255.0f));
            if (sortedLabels.size() > NO_OF_RESULTS) {
                sortedLabels.poll();
            }
        }

        // get top results from priority queue
        for (int i = 0; i < NO_OF_RESULTS; ++i) {
            Map.Entry<String, Float> label = sortedLabels.poll();
            topLabels[i] = label.getKey();
            topConfidence[i] = String.format("%.0f%%",label.getValue()*100);
        }

        // set the corresponding textviews with the results
        txtPred1.setText(topLabels[2] + " " + topConfidence[2]);
        txtPred2.setText(topLabels[1] + " " + topConfidence[1]);
        txtPred3.setText(topLabels[0] + " " + topConfidence[0]);
    }
}