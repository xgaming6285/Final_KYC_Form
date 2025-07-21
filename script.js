// Mobile Navigation Toggle
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.getElementById('hamburger');
    const navMenu = document.getElementById('nav-menu');
    
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
        
        // Update display
        monthlyPayment.textContent = Math.round(monthly) + ' лв';
        interestRate.textContent = baseRate.toFixed(1) + '%';
        totalAmount.textContent = Math.round(total).toLocaleString('bg-BG') + ' лв';
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

// Start Application Function
function startApplication() {
    alert('Пренасочване към формата за кандидатстване...\n\nВ реален сценарий тук ще бъде отворена страницата с формата за кандидатстване.');
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

// Language Selector Functionality
document.addEventListener('DOMContentLoaded', function() {
    const languageSelect = document.getElementById('language-select');
    
    languageSelect.addEventListener('change', function() {
        const selectedLang = this.value;
        
        // In a real application, this would switch the language
        // For demo purposes, we'll just show an alert
        if (selectedLang === 'en') {
            alert('Language switching to English would be implemented here.\n\nВ реален сценарий тук ще се сменя езикът на английски.');
            this.value = 'bg'; // Reset to Bulgarian
        }
    });
});

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

// Scroll animations
function animateOnScroll() {
    const elements = document.querySelectorAll('.step, .testimonial, .faq-item, .contact-item');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    elements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(element);
    });
}

// Initialize scroll animations when DOM is loaded
document.addEventListener('DOMContentLoaded', animateOnScroll);

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