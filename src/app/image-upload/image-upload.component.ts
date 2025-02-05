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
  uploadedImage: string | ArrayBuffer | null = null;
  imageError: string | null = null;
  detectionResultsOriginal: { x: number; y: number; width: number; height: number; label: string; confidence: number; }[] | null = null;
  detectionResultsBrightened: { x: number; y: number; width: number; height: number; label: string; confidence: number; }[] | null = null;
  brightenedImage: string | ArrayBuffer | null = null;
  detectedImageUrl: string | null = null;
  backendUrl = 'http://localhost:5000/';

  constructor(private http: HttpClient) {}

  onFileSelected(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        const formData = new FormData();
        formData.append('image', file);
  
        const reader = new FileReader();
        reader.onload = () => {
          this.uploadedImage = reader.result as string; // Convert to string
          console.log(this.uploadedImage);
          this.performObjectDetection(this.uploadedImage, true); // Detect objects
          this.brighten(this.uploadedImage); // Uncomment if needed
        };
  
        reader.readAsDataURL(file); // Must call this to trigger 'onload'
      } else {
        this.imageError = 'Please upload a valid image file.';
        console.error(this.imageError);
      }
    }
  }
  
  //To call YOLO
  performObjectDetection(base64Image: string, isOriginal: boolean) {
    const requestBody = { image: base64Image };
    console.log("Sending image to detection API:", requestBody);

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

  brighten(base64Image: string) {
    const requestBody = { image: base64Image }; //send as JSON payload

    this.http.post<{ enhanced_image_url: string }>(
      this.backendUrl + 'brighten',
      requestBody
    ).pipe(
      tap((response) => {
        this.brightenedImage = response.enhanced_image_url; // Ensure matches response from API
        this.performObjectDetection(this.brightenedImage as string, false); // Call object detection API on enhanced image  
      }),
      catchError((error) => {
        console.error('Image enhancement failed:', error);
        this.imageError = 'Failed to enhance image. Please try again.';
        return throwError(() => error); // Ensure the observable continues to propagate the error if needed
      })
    ).subscribe();
    
  }
}
