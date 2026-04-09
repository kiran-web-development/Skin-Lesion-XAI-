document.getElementById('imageInput').addEventListener('change', handleImageUpload);

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file
    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showError('Invalid file type. Please upload dermoscopic images only (JPG, PNG, GIF, BMP).');
        return;
    }

    // Hide previous results and show loading
    document.getElementById('results').classList.add('hidden');
    document.getElementById('errorMsg').classList.add('hidden');
    document.getElementById('loading').classList.remove('hidden');

    // Upload and predict
    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        showError('Error: ' + error.message);
    })
    .finally(() => {
        document.getElementById('loading').classList.add('hidden');
    });
}

function displayResults(data) {
    // Display input image
    document.getElementById('inputImage').src = data.input_image;

    // Display prediction
    document.getElementById('predictedClass').textContent = data.predicted_class;
    document.getElementById('confidenceText').textContent = 
        `Confidence: ${data.confidence.toFixed(2)}%`;
    
    // Animate confidence bar
    const confidenceBar = document.getElementById('confidenceBar');
    confidenceBar.style.width = '0%';
    setTimeout(() => {
        confidenceBar.style.width = data.confidence + '%';
    }, 100);

    // Display all probabilities
    const probList = document.getElementById('probabilitiesList');
    probList.innerHTML = '';
    data.all_probabilities.forEach(([className, prob]) => {
        const item = document.createElement('div');
        item.className = 'probability-item';
        item.innerHTML = `
            <span class="class-name">${className}</span>
            <span class="class-prob">${prob.toFixed(2)}%</span>
        `;
        probList.appendChild(item);
    });

    // Display LIME explanation
    document.getElementById('limeExplanation').src = data.lime_explanation;

    // Show results
    document.getElementById('results').classList.remove('hidden');
}

function showError(message) {
    const errorDiv = document.getElementById('errorMsg');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

function resetForm() {
    document.getElementById('imageInput').value = '';
    document.getElementById('results').classList.add('hidden');
    document.getElementById('errorMsg').classList.add('hidden');
    document.getElementById('loading').classList.add('hidden');
}

// Allow drag and drop
const uploadBox = document.querySelector('.upload-box');

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f0f1ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = '#f8f9ff';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        document.getElementById('imageInput').files = files;
        handleImageUpload({ target: { files: files } });
    }
});
