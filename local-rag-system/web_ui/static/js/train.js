// Training functionality
class TrainingInterface {
    constructor() {
        this.trainDocsButton = document.getElementById('trainDocs');
        this.trainImagesButton = document.getElementById('trainImages');
        this.docsProgress = document.getElementById('docsProgress');
        this.imagesProgress = document.getElementById('imagesProgress');
        this.docsResult = document.getElementById('docsResult');
        this.imagesResult = document.getElementById('imagesResult');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        this.trainDocsButton.addEventListener('click', () => this.trainDocuments());
        this.trainImagesButton.addEventListener('click', () => this.trainImages());
    }
    
    async trainDocuments() {
        this.trainDocsButton.disabled = true;
        this.docsProgress.style.display = 'block';
        this.docsResult.innerHTML = '';
        
        try {
            const response = await fetch('/api/train/documents', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.docsResult.innerHTML = `
                    <div class="alert alert-success">
                        <strong>Training Complete!</strong><br>
                        📄 Processed: ${data.processed} documents<br>
                        ❌ Errors: ${data.errors}<br>
                        💾 Total in database: ${data.total_in_db}
                    </div>
                `;
            } else {
                this.docsResult.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Error:</strong> ${data.error}
                    </div>
                `;
            }
        } catch (error) {
            this.docsResult.innerHTML = `
                <div class="alert alert-danger">
                    <strong>Connection Error:</strong> ${error.message}
                </div>
            `;
        } finally {
            this.docsProgress.style.display = 'none';
            this.trainDocsButton.disabled = false;
        }
    }
    
    async trainImages() {
        this.trainImagesButton.disabled = true;
        this.imagesProgress.style.display = 'block';
        this.imagesResult.innerHTML = '';
        
        try {
            const response = await fetch('/api/train/images', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.imagesResult.innerHTML = `
                    <div class="alert alert-success">
                        <strong>Training Complete!</strong><br>
                        🖼️ Processed: ${data.processed} images<br>
                        ❌ Errors: ${data.errors}<br>
                        💾 Total in database: ${data.total_in_db}
                    </div>
                `;
            } else {
                this.imagesResult.innerHTML = `
                    <div class="alert alert-danger">
                        <strong>Error:</strong> ${data.error}
                    </div>
                `;
            }
        } catch (error) {
            this.imagesResult.innerHTML = `
                <div class="alert alert-danger">
                    <strong>Connection Error:</strong> ${error.message}
                </div>
            `;
        } finally {
            this.imagesProgress.style.display = 'none';
            this.trainImagesButton.disabled = false;
        }
    }
}

// Initialize training interface when page loads
document.addEventListener('DOMContentLoaded', () => {
    new TrainingInterface();
});
