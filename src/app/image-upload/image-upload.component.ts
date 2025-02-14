import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { tap, catchError } from 'rxjs/operators';
import { throwError } from 'rxjs';
import { NzCardModule } from 'ng-zorro-antd/card';
import { NzTagModule } from 'ng-zorro-antd/tag';
import { NzFormModule } from 'ng-zorro-antd/form';
import { NzInputModule } from 'ng-zorro-antd/input';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzDatePickerModule } from 'ng-zorro-antd/date-picker';

@Component({
  selector: 'app-image-upload',
  standalone: true,
  imports: [CommonModule, NzFormModule, NzInputModule, NzButtonModule, NzDatePickerModule, NzCardModule, NzTagModule],
  templateUrl: './image-upload.component.html',
  styleUrls: ['./image-upload.component.css']
})
export class ImageUploadComponent {
  uploadedImagePath: string | null = null;
  imageError: string | null = null;
  detectionResultsOriginal: { x: number; y: number; width: number; height: number; label: string; confidence: number; }[] | null = null;
  detectionResultsBrightened: { x: number; y: number; width: number; height: number; label: string; confidence: number; }[] | null = null;
  brightenedImagePath: string | null = null;
  detectedImageUrl: string | null = null;
  backendUrl = 'http://localhost:5000/';
  scaleX: number = 1;
  scaleY: number = 1;

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        const formData = new FormData();
        formData.append('image', file);
        // Send the file to the backend
        this.http.post<{ filePath: string }>(
          this.backendUrl + 'upload', 
          formData
        ).subscribe(
          (response) => {
            this.processFilePath(response, "normal");
            console.log('File saved at:', this.uploadedImagePath);

            if (this.uploadedImagePath) {
              this.performObjectDetection(this.uploadedImagePath, true); // Pass the file path
              this.brighten(this.uploadedImagePath);
            } else {
              console.error('Uploaded image path is null.');
            }
          },
          (error) => {
            console.error('Error uploading file:', error);
            this.imageError = 'Failed to upload the image.';
          }
        );
      } else {
        this.imageError = 'Please upload a valid image file.';
        console.error(this.imageError);
      }
    }
  }

  onImageLoad(event: Event) {
    const image = event.target as HTMLImageElement;
    this.scaleX = image.width / image.naturalWidth;
    this.scaleY = image.height / image.naturalHeight;
  }
  
  //To call YOLO
  performObjectDetection(imagePath: string, isOriginal: boolean) {
    const localFilePath = './backend/uploads/' + imagePath.split('/').pop();
    const requestBody = { filePath: localFilePath };
    console.log("Sending image path to detection API:", requestBody);

    this.http.post<{ detections: { x: number; y: number; width: number; height: number; label: string; confidence: number }[] }>(
      this.backendUrl + 'detect',
      requestBody
    ).pipe(
      tap((response) => {
        if (isOriginal) {
          this.detectionResultsOriginal = response.detections; // Ensure matches response from API
          console.log('Original detection results:', this.detectionResultsOriginal);
        } else {
          this.detectionResultsBrightened = response.detections; // Ensure matches response from API
          console.log('Brightened detection results:', this.detectionResultsBrightened);
        }
      }),
      catchError((error) => {
        console.error('Object detection failed:', error);
        this.imageError = 'Failed to perform object detection. Please try again.';
        return throwError(() => error); // Ensure the observable continues to propagate the error if needed
      })
    ).subscribe();
  }

  brighten(imagePath: string) {
    const localFilePath = './backend/uploads/'  + imagePath.split('/').pop();
    const requestBody = { filePath: localFilePath };
    console.log("Sending image path to brighten API:", requestBody);

    this.http.post<{ enhanced_image_path: string }>(
      this.backendUrl + 'brighten',
      requestBody
    ).pipe(
      tap((response) => {
        this.processFilePath(response, "brightened");
        console.log("sending brightened image path to detection API: ,", this.brightenedImagePath);
        if (this.brightenedImagePath) {
          this.performObjectDetection(this.brightenedImagePath, false); // Call object detection API on enhanced image  
        } else {
          console.error('Brightened image path is null.');
        }
      }),
      catchError((error) => {
        console.error('Image enhancement failed:', error);
        this.imageError = 'Failed to enhance image. Please try again.';
        return throwError(() => error); // Ensure the observable continues to propagate the error if needed
      })
    ).subscribe();
  }

  processFilePath(response: { enhanced_image_path?: string; filePath?: string }, variable: string) {
    if (variable === "brightened") {
      const currPath = response.enhanced_image_path;
      if (currPath) {
        this.brightenedImagePath = this.backendUrl + 'uploads/' + currPath.split('/').pop();
      }
    } else if (variable === "normal") {
      const currPath = response.filePath;
      if (currPath) {
        this.uploadedImagePath = this.backendUrl + 'uploads/' + currPath.split('/').pop();
      }
    }
  }  
}
