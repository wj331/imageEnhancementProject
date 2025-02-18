import { Component, ChangeDetectorRef} from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { EMPTY, throwError , from, forkJoin} from 'rxjs';
import { NzCardModule } from 'ng-zorro-antd/card';
import { NzTagModule } from 'ng-zorro-antd/tag';
import { NzFormModule } from 'ng-zorro-antd/form';
import { NzInputModule } from 'ng-zorro-antd/input';
import { NzButtonModule } from 'ng-zorro-antd/button';
import { NzDatePickerModule } from 'ng-zorro-antd/date-picker';
import { concatMap, tap, catchError, switchMap, map} from 'rxjs/operators';

@Component({
  selector: 'app-image-upload',
  standalone: true,
  imports: [CommonModule, NzFormModule, NzInputModule, NzButtonModule, NzDatePickerModule, NzCardModule, NzTagModule],
  templateUrl: './image-upload.component.html',
  styleUrls: ['./image-upload.component.css']
})
export class ImageUploadComponent {

  uploadedImages: { path: string, detectionResults: any[] }[] = [];
  brightenedImages: { path: string, detectionResults: any[] }[] = [];
  imageError: string | null = null;
  detectionResultsOriginal: { x: number; y: number; width: number; height: number; label: string; confidence: number; }[] | null = null;
  detectionResultsBrightened: { x: number; y: number; width: number; height: number; label: string; confidence: number; }[] | null = null;
  brightenedImagePath: string | null = null;
  uploadedImagePath: string | null = null;
  detectedImageUrl: string | null = null;
  backendUrl = 'http://localhost:5000/';
  scaleX: number = 1;
  scaleY: number = 1;

  constructor(private http: HttpClient, private cdRef: ChangeDetectorRef) {}
    onFileSelected(event: Event) {
      const files = (event.target as HTMLInputElement).files;
    
      if (files && files.length > 0) {
        const imageFiles = Array.from(files).filter(file => file.type.startsWith('image/'));
    
        if (imageFiles.length === 0) {
          this.imageError = 'No valid image files found in the selected folder.';
          console.error(this.imageError);
          return;
        }
    
        from(imageFiles).pipe(
          concatMap((file) => this.processImage(file).pipe(
            catchError((error) => {
              console.error('Error during file processing:', error);
              return EMPTY;  // Continue processing other files even if one fails
            })
          ))
        ).subscribe({
          complete: () =>  console.log('All files processed successfully.')
        });
      } else {
        this.imageError = 'No files selected.';
        console.error(this.imageError);
      }
    }
    
    processImage(file: File) {
      const formData = new FormData();
      formData.append('image', file);
        
      // Step-by-step processing pipeline
      return this.http.post<{ filePath: string }>(`${this.backendUrl}upload`, formData).pipe(
        switchMap(response => {
          console.log('File uploaded', response.filePath);
          this.uploadedImages.push({ path: response.filePath, detectionResults: [] });
        return forkJoin({
          originalDetection: this.performObjectDetection(response.filePath, true),
          brightenedPath: this.brighten(response.filePath)
        }).pipe(
          switchMap(({ originalDetection, brightenedPath }) => {
            console.log('Brightened image path:', brightenedPath);
            
            this.brightenedImages.push({ path: brightenedPath, detectionResults: []});

            return this.performObjectDetection(brightenedPath, false).pipe(
              tap(brightenedDetection => {
                console.log('Detection results for brightened image:', brightenedDetection);
              })
            );
          })
        );
      }),
      catchError(error => {
        console.error('Error processing image:', file.name, error);
        return EMPTY;
      })
    )
  }
    
    performObjectDetection(imagePath: string, isOriginal: boolean) {
      const requestBody = { filePath: imagePath };
    
      return this.http.post<{ detections: { x: number, y: number, width: number, height: number, label: string, confidence: number }[] }>(
        `${this.backendUrl}detect`,
        requestBody
      ).pipe(
        tap((response) => {
          if (isOriginal) {
            const image = this.uploadedImages.find(img => img.path === imagePath);
            if (image) {
              image.detectionResults = response.detections;
              this.cdRef.detectChanges();
              console.log('Original detection results:', this.detectionResultsOriginal);
            }
          } else {
            const image = this.brightenedImages.find(img => img.path === imagePath);
            if (image) {
              image.detectionResults = response.detections;
              this.cdRef.detectChanges();
              console.log('Brightened detection results:', this.detectionResultsBrightened);
            }
          }
        })
      );
    }
    
    brighten(imagePath: string) {
      const requestBody = { filePath: imagePath };
    
      return this.http.post<{ enhanced_image_path: string }>(
        `${this.backendUrl}brighten`,
        requestBody
      ).pipe(
        map((response : {enhanced_image_path: string}) => response.enhanced_image_path),  // Return the path of the brightened image
        tap((brightenedPath) => console.log('Brightened image url:', brightenedPath)),
        catchError((error) => {
          console.error('Error during image brightening:', error);
          return EMPTY;
        })
      );
    }
}
