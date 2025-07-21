// Enhanced Mobile Navigation Toggle with Modern Features
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');
    const navbar = document.querySelector('.navbar');
    const scrollProgress = document.getElementById('scroll-progress');
    const scrollToTopBtn = document.getElementById('scroll-to-top');
    
    // Mobile menu toggle
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });

        // Close mobile menu when clicking on a link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }
    
    // Enhanced scroll effects
    function updateScrollEffects() {
        const scrolled = window.scrollY;
        const documentHeight = document.documentElement.scrollHeight - window.innerHeight;
        const progress = (scrolled / documentHeight) * 100;
        
        // Update scroll progress bar
        if (scrollProgress) {
            scrollProgress.style.width = `${progress}%`;
        }
        
        // Update navbar on scroll
        if (navbar) {
            if (scrolled > 100) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        }
        
        // Show/hide scroll to top button
        if (scrollToTopBtn) {
            if (scrolled > 300) {
                scrollToTopBtn.classList.add('visible');
            } else {
                scrollToTopBtn.classList.remove('visible');
            }
        }
    }
    
    // Smooth scroll to top
    if (scrollToTopBtn) {
        scrollToTopBtn.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }
    
    // Throttled scroll event listener for better performance
    let scrollTimeout;
    window.addEventListener('scroll', function() {
        if (!scrollTimeout) {
            scrollTimeout = setTimeout(function() {
                updateScrollEffects();
                scrollTimeout = null;
            }, 10);
        }
    });
    
    // Initialize scroll effects
    updateScrollEffects();
    
    // Add page transition effect
    document.body.classList.add('page-transition');
});

// Smooth Scrolling for Navigation Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Scroll to Calculator Function
function scrollToCalculator() {
    document.getElementById('calculator').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Loan Calculator Functionality
document.addEventListener('DOMContentLoaded', function() {
    const loanAmountSlider = document.getElementById('loan-amount');
    const loanPeriodSlider = document.getElementById('loan-period');
    const amountValue = document.getElementById('amount-value');
    const periodValue = document.getElementById('period-value');
    const monthlyPayment = document.getElementById('monthly-payment');
    const interestRate = document.getElementById('interest-rate');
    const totalAmount = document.getElementById('total-amount');

    // Update display values and calculations
    function updateCalculator() {
        const amount = parseInt(loanAmountSlider.value);
        const period = parseInt(loanPeriodSlider.value);
        
        // Format amount display
        amountValue.textContent = amount.toLocaleString('bg-BG');
        periodValue.textContent = period;

        // Calculate loan details
        const baseRate = calculateInterestRate(amount, period);
        const monthlyRate = baseRate / 100 / 12;
        const numberOfPayments = period;
        
        // Calculate monthly payment using loan formula
        const monthly = (amount * monthlyRate * Math.pow(1 + monthlyRate, numberOfPayments)) / 
                       (Math.pow(1 + monthlyRate, numberOfPayments) - 1);
        
        const total = monthly * numberOfPayments;
        
        // Update display with proper currency
        const currency = translationManager ? translationManager.t('calculator.currency') : 'лв';
        const locale = translationManager && translationManager.getCurrentLanguage() === 'en' ? 'en-US' : 'bg-BG';
        
        monthlyPayment.textContent = Math.round(monthly) + ' ' + currency;
        interestRate.textContent = baseRate.toFixed(1) + '%';
        totalAmount.textContent = Math.round(total).toLocaleString(locale) + ' ' + currency;
    }

    // Calculate interest rate based on amount and period
    function calculateInterestRate(amount, period) {
        let baseRate = 12.5; // Base rate
        
        // Adjust rate based on amount (lower rate for higher amounts)
        if (amount >= 30000) baseRate -= 1.5;
        else if (amount >= 20000) baseRate -= 1.0;
        else if (amount >= 10000) baseRate -= 0.5;
        
        // Adjust rate based on period (lower rate for longer periods)
        if (period >= 48) baseRate -= 1.0;
        else if (period >= 36) baseRate -= 0.5;
        else if (period <= 12) baseRate += 1.0;
        
        return Math.max(8.5, Math.min(18.0, baseRate)); // Keep rate between 8.5% and 18%
    }

    // Add event listeners
    loanAmountSlider.addEventListener('input', updateCalculator);
    loanPeriodSlider.addEventListener('input', updateCalculator);

    // Initial calculation
    updateCalculator();
});

// Start Application Function - REDIRECT TO VERIFICATION PAGE
function startApplication() {
    window.location.href = 'verification.html';
}


// FAQ Toggle Functionality
function toggleFAQ(index) {
    const faqItems = document.querySelectorAll('.faq-item');
    const currentItem = faqItems[index];
    
    // Close all other FAQs
    faqItems.forEach((item, i) => {
        if (i !== index) {
            item.classList.remove('active');
        }
    });
    
    // Toggle current FAQ
    currentItem.classList.toggle('active');
}

// Testimonials Slider Functionality
let currentTestimonial = 0;
const testimonials = document.querySelectorAll('.testimonial');

function showTestimonial(index) {
    testimonials.forEach((testimonial, i) => {
        testimonial.classList.toggle('active', i === index);
    });
}

function changeTestimonial(direction) {
    currentTestimonial += direction;
    
    if (currentTestimonial >= testimonials.length) {
        currentTestimonial = 0;
    } else if (currentTestimonial < 0) {
        currentTestimonial = testimonials.length - 1;
    }
    
    showTestimonial(currentTestimonial);
}

// Auto-rotate testimonials
setInterval(() => {
    changeTestimonial(1);
}, 5000);

// Help Modal Functionality
function toggleHelp() {
    const helpModal = document.getElementById('help-modal');
    helpModal.classList.toggle('active');
}

// Close modal when clicking outside
document.addEventListener('click', function(e) {
    const helpModal = document.getElementById('help-modal');
    const helpButton = document.querySelector('.help-button');
    
    if (helpModal.classList.contains('active') && 
        !helpModal.querySelector('.help-content').contains(e.target) && 
        !helpButton.contains(e.target)) {
        helpModal.classList.remove('active');
    }
});

// Translation System Initialization (Opt-in only)
let translationManager;

document.addEventListener('DOMContentLoaded', async function() {
    try {
        // Initialize translation system (doesn't translate anything automatically)
        translationManager = new TranslationManager();
        await translationManager.init();
        
        console.log('Translation system initialized successfully');
        
        // Add observer for language changes (optional - for custom behaviors)
        translationManager.addObserver((language) => {
            console.log(`Language changed to: ${language}`);
            
            // Custom behaviors on language change can be added here
            // For example, reformat numbers, dates, etc.
            updateNumberFormats(language);
        });
    } catch (error) {
        console.log('Translation system failed to load, site will work without translations');
    }
});

// Update number formats based on language
function updateNumberFormats(language) {
    const locale = language === 'bg' ? 'bg-BG' : 'en-US';
    
    // Update loan calculator display
    const amountValue = document.getElementById('amount-value');
    const totalAmount = document.getElementById('total-amount');
    
    if (amountValue) {
        const amount = parseInt(amountValue.textContent.replace(/[^\d]/g, ''));
        amountValue.textContent = amount.toLocaleString(locale);
    }
    
    if (totalAmount) {
        const total = parseInt(totalAmount.textContent.replace(/[^\d]/g, ''));
        if (!isNaN(total)) {
            const currency = translationManager.t('calculator.currency');
            totalAmount.textContent = `${total.toLocaleString(locale)} ${currency}`;
        }
    }
}

// Form Validation and Enhancement
document.addEventListener('DOMContentLoaded', function() {
    // Add loading states to buttons
    const buttons = document.querySelectorAll('.cta-button, .apply-button');
    
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (!this.classList.contains('loading')) {
                this.classList.add('loading');
                const originalText = this.innerHTML;
                this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Зареждане...';
                
                setTimeout(() => {
                    this.classList.remove('loading');
                    this.innerHTML = originalText;
                }, 2000);
            }
        });
    });
});

// Enhanced Button Interactions
document.addEventListener('DOMContentLoaded', function() {
    // Add enhanced button effects
    const buttons = document.querySelectorAll('.cta-button, .apply-button, .btn');
    
    buttons.forEach(button => {
        button.classList.add('btn-enhanced');
        
        // Add ripple effect on click
        button.addEventListener('click', function(e) {
            const ripple = document.createElement('span');
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = e.clientX - rect.left - size / 2;
            const y = e.clientY - rect.top - size / 2;
            
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = x + 'px';
            ripple.style.top = y + 'px';
            ripple.classList.add('ripple');
            
            this.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        });
    });
    
    // Enhanced form interactions
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.classList.add('form-enhanced');
    });
    
    // Add floating label effect
    const inputs = document.querySelectorAll('input, select, textarea');
    inputs.forEach(input => {
        // Add focus and blur effects
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });
        
        // Initialize state
        if (input.value) {
            input.parentElement.classList.add('focused');
        }
    });
    
    // Add hover effects to cards
    const cards = document.querySelectorAll('.option-card, .result-card, .step, .testimonial, .faq-item');
    cards.forEach(card => {
        card.classList.add('hover-lift');
    });
    
    // Add animated icons
    const icons = document.querySelectorAll('.feature i, .nav-logo i, .hero-placeholder i');
    icons.forEach(icon => {
        icon.classList.add('icon-animated');
    });
    
    // Enhanced loading states
    const loadingButtons = document.querySelectorAll('[data-loading]');
    loadingButtons.forEach(button => {
        button.addEventListener('click', function() {
            this.classList.add('loading');
            
            // Remove loading state after animation
            setTimeout(() => {
                this.classList.remove('loading');
            }, 2000);
        });
    });
});

// Enhanced Scroll Animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.animationPlayState = 'running';
            entry.target.classList.add('animate-in');
        }
    });
}, observerOptions);

// Observe elements for scroll animations
document.addEventListener('DOMContentLoaded', function() {
    const animatedElements = document.querySelectorAll(
        '.hero-content > *, .calculator-section > *, .feature, .step, .testimonial'
    );
    
    animatedElements.forEach(element => {
        element.style.animationPlayState = 'paused';
        observer.observe(element);
    });
});

// Navbar scroll effect
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'rgba(255, 255, 255, 0.95)';
        navbar.style.backdropFilter = 'blur(10px)';
    } else {
        navbar.style.background = '#ffffff';
        navbar.style.backdropFilter = 'none';
    }
});

// Counter animation for hero features
function animateCounters() {
    const counters = document.querySelectorAll('.hero-content .highlight');
    
    counters.forEach(counter => {
        const target = parseInt(counter.textContent.replace(/[^\d]/g, ''));
        const duration = 2000;
        const increment = target / (duration / 50);
        let current = 0;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            counter.textContent = Math.floor(current).toLocaleString('bg-BG') + ' лв';
        }, 50);
    });
}

// Start counter animation when hero section is visible
const heroObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            animateCounters();
            heroObserver.unobserve(entry.target);
        }
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const hero = document.querySelector('.hero');
    if (hero) {
        heroObserver.observe(hero);
    }
});

// Enhanced slider functionality with touch support
document.addEventListener('DOMContentLoaded', function() {
    const sliders = document.querySelectorAll('input[type="range"]');
    
    sliders.forEach(slider => {
        // Create custom slider track fill
        const updateSliderFill = () => {
            const percent = (slider.value - slider.min) / (slider.max - slider.min) * 100;
            slider.style.background = `linear-gradient(to right, #1e40af 0%, #1e40af ${percent}%, #e5e7eb ${percent}%, #e5e7eb 100%)`;
        };
        
        slider.addEventListener('input', updateSliderFill);
        updateSliderFill(); // Initial setup
    });
});

// Performance optimization - lazy loading for images
document.addEventListener('DOMContentLoaded', function() {
    const lazyImages = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.removeAttribute('data-src');
                imageObserver.unobserve(img);
            }
        });
    });
    
    lazyImages.forEach(img => imageObserver.observe(img));
});

// Error handling and user feedback
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
    // In production, you might want to send this to a logging service
});

// Print functionality (for loan calculations)
function printLoanDetails() {
    const amount = document.getElementById('amount-value').textContent;
    const period = document.getElementById('period-value').textContent;
    const monthly = document.getElementById('monthly-payment').textContent;
    const rate = document.getElementById('interest-rate').textContent;
    const total = document.getElementById('total-amount').textContent;
    
    const printContent = `
        <h2>Детайли на кредита</h2>
        <p><strong>Сума:</strong> ${amount} лв</p>
        <p><strong>Период:</strong> ${period} месеца</p>
        <p><strong>Месечна вноска:</strong> ${monthly}</p>
        <p><strong>Лихвен процент:</strong> ${rate}</p>
        <p><strong>Общо за връщане:</strong> ${total}</p>
        <p><em>Генерирано на ${new Date().toLocaleDateString('bg-BG')}</em></p>
    `;
    
    const printWindow = window.open('', '_blank');
    printWindow.document.write(`
        <html>
            <head>
                <title>Детайли на кредита</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 20px; }
                    h2 { color: #1e40af; }
                    p { margin: 10px 0; }
                </style>
            </head>
            <body>${printContent}</body>
        </html>
    `);
    printWindow.document.close();
    printWindow.print();
}

// Add print button to calculator (optional)
document.addEventListener('DOMContentLoaded', function() {
    const resultCard = document.querySelector('.result-card');
    if (resultCard) {
        const printButton = document.createElement('button');
        printButton.innerHTML = '<i class="fas fa-print"></i> Принтирай детайлите';
        printButton.className = 'print-button';
        printButton.style.cssText = `
            background: #6b7280;
            color: white;
            border: none;
            padding: 12px 20px;
            border-radius: 6px;
            font-size: 14px;
            cursor: pointer;
            margin-top: 16px;
            width: 100%;
            transition: background 0.3s ease;
        `;
        printButton.addEventListener('mouseenter', () => {
            printButton.style.background = '#4b5563';
        });
        printButton.addEventListener('mouseleave', () => {
            printButton.style.background = '#6b7280';
        });
        printButton.addEventListener('click', printLoanDetails);
        resultCard.appendChild(printButton);
    }
});

// Accessibility improvements
document.addEventListener('DOMContentLoaded', function() {
    // Add keyboard navigation for sliders
    const sliders = document.querySelectorAll('input[type="range"]');
    sliders.forEach(slider => {
        slider.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                e.preventDefault();
                const step = parseInt(this.step) || 1;
                const change = e.key === 'ArrowRight' ? step : -step;
                const newValue = Math.min(Math.max(parseInt(this.value) + change, parseInt(this.min)), parseInt(this.max));
                this.value = newValue;
                this.dispatchEvent(new Event('input'));
            }
        });
    });

    // Add ARIA labels for better accessibility
    document.querySelectorAll('.faq-question').forEach((question, index) => {
        question.setAttribute('role', 'button');
        question.setAttribute('aria-expanded', 'false');
        question.setAttribute('aria-controls', `faq-answer-${index}`);
        
        const answer = question.nextElementSibling;
        if (answer) {
            answer.setAttribute('id', `faq-answer-${index}`);
            answer.setAttribute('role', 'region');
        }
    });
});

// ========== VERIFICATION FORM FUNCTIONALITY ==========

let currentStep = 1;
let uploadedFiles = {
    idFront: null,
    idBack: null,
    incomeDoc: null,
    facePhoto: null
};
let currentCameraTarget = null;
let faceStream = null;

// Close Verification Modal
function closeVerification() {
    const verificationModal = document.getElementById('verification-modal');
    verificationModal.classList.remove('active');
    document.body.style.overflow = 'auto';
    
    // Reset form
    resetVerificationForm();
    
    // Stop any active camera streams
    stopAllStreams();
}

// Reset verification form to initial state
function resetVerificationForm() {
    currentStep = 1;
    uploadedFiles = {
        idFront: null,
        idBack: null,
        incomeDoc: null,
        facePhoto: null
    };
    
    // Reset step indicators
    document.querySelectorAll('.step-item').forEach((item, index) => {
        item.classList.remove('active', 'completed');
        if (index === 0) item.classList.add('active');
    });
    
    // Reset steps
    document.querySelectorAll('.verification-step').forEach((step, index) => {
        step.classList.remove('active');
        if (index === 0) step.classList.add('active');
    });
    
    // Clear form fields
    document.getElementById('personal-info-form').reset();
    
    // Clear previews
    clearAllPreviews();
}

// Step Navigation
function nextStep(stepNumber) {
    if (validateCurrentStep()) {
        // Mark current step as completed
        const currentStepItem = document.querySelector(`.step-item[data-step="${currentStep}"]`);
        if (currentStepItem) {
            currentStepItem.classList.remove('active');
            currentStepItem.classList.add('completed');
        }
        
        // Show next step
        document.getElementById(`step-${currentStep}`).classList.remove('active');
        document.getElementById(`step-${stepNumber}`).classList.add('active');
        
        // Update step indicator
        const nextStepItem = document.querySelector(`.step-item[data-step="${stepNumber}"]`);
        if (nextStepItem) {
            nextStepItem.classList.add('active');
        }
        
        currentStep = stepNumber;
    }
}

function prevStep(stepNumber) {
    // Mark current step as inactive
    const currentStepItem = document.querySelector(`.step-item[data-step="${currentStep}"]`);
    if (currentStepItem) {
        currentStepItem.classList.remove('active');
    }
    
    // Show previous step
    document.getElementById(`step-${currentStep}`).classList.remove('active');
    document.getElementById(`step-${stepNumber}`).classList.add('active');
    
    // Update step indicator
    const prevStepItem = document.querySelector(`.step-item[data-step="${stepNumber}"]`);
    if (prevStepItem) {
        prevStepItem.classList.remove('completed');
        prevStepItem.classList.add('active');
    }
    
    currentStep = stepNumber;
}

// Validation
function validateCurrentStep() {
    switch (currentStep) {
        case 1:
            return validatePersonalInfo();
        case 2:
            return validateDocuments();
        case 3:
            return validateFaceVerification();
        default:
            return true;
    }
}

function validatePersonalInfo() {
    const form = document.getElementById('personal-info-form');
    const formData = new FormData(form);
    
    // Basic validation
    for (let [key, value] of formData.entries()) {
        if (!value.trim()) {
            alert('Моля попълнете всички задължителни полета.');
            return false;
        }
    }
    
    // EGN validation (basic)
    const egn = formData.get('egn');
    if (egn && egn.length !== 10) {
        alert('ЕГН трябва да бъде 10 цифри.');
        return false;
    }
    
    return true;
}

function validateDocuments() {
    if (!uploadedFiles.idFront || !uploadedFiles.idBack) {
        alert('Моля качете снимки от двете страни на личната карта.');
        return false;
    }
    
    if (!uploadedFiles.incomeDoc) {
        alert('Моля качете документ за доходи.');
        return false;
    }
    
    return true;
}

function validateFaceVerification() {
    if (!uploadedFiles.facePhoto) {
        alert('Моля направете селфи за верификация на лице.');
        return false;
    }
    
    return true;
}

// File Upload Handling
document.addEventListener('DOMContentLoaded', function() {
    // Setup file upload areas
    setupFileUpload('id-front', 'id-front-preview');
    setupFileUpload('id-back', 'id-back-preview');
    setupFileUpload('income-doc', 'income-preview');
    setupFileUpload('face-upload', 'face-preview');
    
    // Setup drag and drop
    setupDragAndDrop();
});

function setupFileUpload(inputId, previewId) {
    const input = document.getElementById(inputId);
    const uploadArea = document.getElementById(inputId.replace('-', '-') + '-upload') || input.parentElement;
    const preview = document.getElementById(previewId);
    
    if (uploadArea && uploadArea !== input.parentElement) {
        uploadArea.addEventListener('click', () => {
            input.click();
        });
    }
    
    input.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleFileUpload(file, inputId, preview);
        }
    });
}

function handleFileUpload(file, inputId, preview) {
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
    if (inputId === 'income-doc') {
        validTypes.push('application/pdf');
    }
    
    if (!validTypes.includes(file.type)) {
        alert('Моля изберете валиден файл (JPG, PNG, WEBP' + (inputId === 'income-doc' ? ', PDF' : '') + ').');
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('Файлът е твърде голям. Максимален размер: 10MB.');
        return;
    }
    
    // Store file
    const fileKey = inputId.replace('-', '');
    uploadedFiles[fileKey] = file;
    
    // Show preview
    if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Преглед на файл">`;
        };
        reader.readAsDataURL(file);
    } else {
        preview.innerHTML = `<p><i class="fas fa-file-pdf"></i> ${file.name}</p>`;
    }
}

function setupDragAndDrop() {
    const uploadAreas = document.querySelectorAll('.upload-area');
    
    uploadAreas.forEach(area => {
        area.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.add('dragover');
        });
        
        area.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('dragover');
        });
        
        area.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            this.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const input = this.querySelector('input[type="file"]');
                if (input) {
                    const inputId = input.id;
                    const previewId = inputId + '-preview';
                    const preview = document.getElementById(previewId);
                    handleFileUpload(files[0], inputId, preview);
                }
            }
        });
    });
}

// Camera Functionality
function openCamera(target) {
    currentCameraTarget = target;
    const cameraModal = document.getElementById('camera-modal');
    cameraModal.classList.add('active');
    
    startCamera();
}

function startCamera() {
    const video = document.getElementById('camera-video');
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'environment', // Use back camera if available
            width: { ideal: 1920 },
            height: { ideal: 1080 }
        } 
    })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(error) {
        console.error('Error accessing camera:', error);
        alert('Не може да се достъпи камерата. Моля използвайте опцията за качване на файл.');
        closeCameraModal();
    });
}

function captureDocument() {
    const video = document.getElementById('camera-video');
    const canvas = document.getElementById('camera-canvas');
    const context = canvas.getContext('2d');
    
    // Set canvas dimensions to video dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert to blob
    canvas.toBlob(function(blob) {
        // Create a file-like object
        const file = new File([blob], `document-${Date.now()}.jpg`, { type: 'image/jpeg' });
        
        // Handle the captured image
        const previewId = currentCameraTarget + '-preview';
        const preview = document.getElementById(previewId);
        handleFileUpload(file, currentCameraTarget, preview);
        
        closeCameraModal();
    }, 'image/jpeg', 0.9);
}

function closeCameraModal() {
    const cameraModal = document.getElementById('camera-modal');
    const video = document.getElementById('camera-video');
    
    // Stop camera stream
    if (video.srcObject) {
        const stream = video.srcObject;
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
    
    cameraModal.classList.remove('active');
    currentCameraTarget = null;
}

// Face Verification
function startFaceCapture() {
    const video = document.getElementById('face-video');
    const placeholder = document.getElementById('face-placeholder');
    const captureBtn = document.querySelector('.capture-btn');
    
    navigator.mediaDevices.getUserMedia({ 
        video: { 
            facingMode: 'user', // Use front camera
            width: { ideal: 640 },
            height: { ideal: 480 }
        } 
    })
    .then(function(stream) {
        faceStream = stream;
        video.srcObject = stream;
        video.style.display = 'block';
        placeholder.style.display = 'none';
        captureBtn.style.display = 'inline-flex';
        video.play();
    })
    .catch(function(error) {
        console.error('Error accessing camera:', error);
        alert('Не може да се достъпи камерата. Моля използвайте опцията за качване на снимка.');
    });
}

function capturePhoto() {
    const video = document.getElementById('face-video');
    const canvas = document.getElementById('face-canvas');
    const context = canvas.getContext('2d');
    const preview = document.getElementById('face-preview');
    
    // Set canvas dimensions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Stop video stream
    if (faceStream) {
        faceStream.getTracks().forEach(track => track.stop());
        faceStream = null;
    }
    
    video.style.display = 'none';
    document.getElementById('face-placeholder').style.display = 'block';
    document.querySelector('.capture-btn').style.display = 'none';
    
    // Convert to blob and store
    canvas.toBlob(function(blob) {
        const file = new File([blob], `face-${Date.now()}.jpg`, { type: 'image/jpeg' });
        uploadedFiles.facePhoto = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.innerHTML = `<img src="${e.target.result}" alt="Селфи за верификация">`;
        };
        reader.readAsDataURL(file);
    }, 'image/jpeg', 0.9);
}

function uploadFaceImage() {
    document.getElementById('face-upload').click();
}

// Clear previews
function clearAllPreviews() {
    const previews = ['id-front-preview', 'id-back-preview', 'income-preview', 'face-preview'];
    previews.forEach(previewId => {
        const preview = document.getElementById(previewId);
        if (preview) {
            preview.innerHTML = '';
        }
    });
}

// Stop all camera streams
function stopAllStreams() {
    // Stop face verification stream
    if (faceStream) {
        faceStream.getTracks().forEach(track => track.stop());
        faceStream = null;
    }
    
    // Stop camera modal stream
    const cameraVideo = document.getElementById('camera-video');
    if (cameraVideo && cameraVideo.srcObject) {
        const stream = cameraVideo.srcObject;
        stream.getTracks().forEach(track => track.stop());
        cameraVideo.srcObject = null;
    }
}

// Submit Application
function submitApplication() {
    if (!validateFaceVerification()) {
        return;
    }
    
    // Show loading state
    const submitBtn = document.querySelector('.submit-btn');
    const originalText = submitBtn.innerHTML;
    const loadingText = translationManager ? translationManager.t('verification.loading', 'Изпращане...') : 'Изпращане...';
    submitBtn.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${loadingText}`;
    submitBtn.disabled = true;
    
    // Simulate application submission
    setTimeout(() => {
        // Hide current step
        document.getElementById('step-3').classList.remove('active');
        
        // Show success step
        document.getElementById('step-success').style.display = 'block';
        
        // Update step indicator - mark all as completed
        document.querySelectorAll('.step-item').forEach(item => {
            item.classList.remove('active');
            item.classList.add('completed');
        });
        
        // Reset button
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
        
        // In a real application, you would send the data to your server here
        console.log('Application submitted with files:', uploadedFiles);
        
        // Stop all streams
        stopAllStreams();
    }, 3000);
}

// Close modal when clicking outside
document.addEventListener('click', function(e) {
    const verificationModal = document.getElementById('verification-modal');
    const cameraModal = document.getElementById('camera-modal');
    
    if (verificationModal.classList.contains('active') && 
        e.target === verificationModal) {
        closeVerification();
    }
    
    if (cameraModal.classList.contains('active') && 
        e.target === cameraModal) {
        closeCameraModal();
    }
});

// Handle modal close with Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const verificationModal = document.getElementById('verification-modal');
        const cameraModal = document.getElementById('camera-modal');
        
        if (cameraModal.classList.contains('active')) {
            closeCameraModal();
        } else if (verificationModal.classList.contains('active')) {
            closeVerification();
        }
    }
}); 