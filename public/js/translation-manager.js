/**
 * Modern Translation Manager
 * Handles internationalization for the website
 */
class TranslationManager {
    constructor() {
        this.currentLanguage = this.getStoredLanguage() || 'bg';
        this.translations = {};
        this.observers = [];
    }

    /**
     * Initialize the translation system
     */
    async init() {
        try {
            await this.loadTranslations();
            this.setupLanguageSelector();
            // Don't update page automatically - only when user explicitly changes language
            this.observeLanguageChanges();
        } catch (error) {
            console.error('Failed to initialize translation system:', error);
            // Fallback to default behavior without translations
        }
    }

    /**
     * Load translations for all supported languages
     */
    async loadTranslations() {
        const languages = ['bg', 'en'];
        
        for (const lang of languages) {
            try {
                const response = await fetch(`translations/${lang}.json`);
                if (response.ok) {
                    this.translations[lang] = await response.json();
                } else {
                    console.warn(`Failed to load ${lang} translations`);
                }
            } catch (error) {
                console.error(`Error loading ${lang} translations:`, error);
            }
        }
    }

    /**
     * Get stored language from localStorage
     */
    getStoredLanguage() {
        return localStorage.getItem('preferred-language');
    }

    /**
     * Store language preference in localStorage
     */
    storeLanguage(language) {
        localStorage.setItem('preferred-language', language);
    }

    /**
     * Setup language selector event listener
     */
    setupLanguageSelector() {
        const languageSelect = document.getElementById('language-select');
        if (languageSelect) {
            // Set current language in dropdown
            languageSelect.value = this.currentLanguage;
            
            // Listen for language changes
            languageSelect.addEventListener('change', (e) => {
                this.changeLanguage(e.target.value);
            });
        }
    }

    /**
     * Change the current language
     */
    async changeLanguage(language) {
        if (this.translations[language]) {
            this.currentLanguage = language;
            this.storeLanguage(language);
            
            // Update all major sections safely
            this.updateNavigationSafe();
            this.updateHeroSafe();
            this.updateCalculatorExpandedSafe();
            this.updateHowItWorksSafe();
            this.updateTestimonialsSafe();
            this.updateSuccessStorySafe();
            this.updateFAQSafe();
            this.updateContactSafe();
            this.updateFooterSafe();
            this.updateHelpSafe();
            this.updateVerificationModalSafe();
            this.updateCameraModalSafe();
            this.updateMiscellaneousSafe();
            
            this.notifyObservers(language);
        } else {
            console.warn(`Language '${language}' not supported`);
        }
    }

    /**
     * Get translation by key path (supports nested keys like 'hero.title')
     */
    t(keyPath, defaultValue = '') {
        if (!this.translations[this.currentLanguage]) {
            return defaultValue;
        }

        const keys = keyPath.split('.');
        let value = this.translations[this.currentLanguage];

        for (const key of keys) {
            if (value && typeof value === 'object' && key in value) {
                value = value[key];
            } else {
                return defaultValue;
            }
        }

        return value || defaultValue;
    }

    /**
     * Update the entire page language
     */
    updatePageLanguage() {
        if (!this.translations[this.currentLanguage]) {
            console.warn(`No translations available for language: ${this.currentLanguage}`);
            return;
        }

        // Update HTML lang attribute
        document.documentElement.lang = this.currentLanguage;

        // Update page title only if translation exists
        const newTitle = this.t('meta.title');
        if (newTitle) {
            document.title = newTitle;
        }

        // Update all elements with data-translate attributes
        this.updateTranslatedElements();

        // Only update safe elements and only when explicitly changing language
        // This prevents overwriting original content on page load
    }

    /**
     * Update elements with data-translate attributes
     */
    updateTranslatedElements() {
        const translatableElements = document.querySelectorAll('[data-translate]');
        
        translatableElements.forEach(element => {
            const key = element.getAttribute('data-translate');
            const translation = this.t(key);
            
            if (translation) {
                if (element.getAttribute('data-translate-html') === 'true') {
                    element.innerHTML = translation;
                } else {
                    element.textContent = translation;
                }
            }
        });
    }

    /**
     * Update elements with special translation patterns
     */
    updateSpecialElements() {
        try {
            // Only update key elements safely
            this.updateNavigationSafe();
            this.updateCalculatorSafe();
        } catch (error) {
            console.error('Error updating special elements:', error);
        }
    }

    /**
     * Update navigation menu safely
     */
    updateNavigationSafe() {
        const navLinks = {
            'a[href="#home"]': 'navigation.home',
            'a[href="#calculator"]': 'navigation.calculator',
            'a[href="#how-it-works"]': 'navigation.how_it_works',
            'a[href="#contact"]': 'navigation.contact'
        };

        Object.entries(navLinks).forEach(([selector, key]) => {
            const element = document.querySelector(selector);
            const translation = this.t(key);
            if (element && translation) {
                element.textContent = translation;
            }
        });
    }

    /**
     * Safely update calculator section only
     */
    updateCalculatorSafe() {
        // Only update specific calculator elements that are safe
        const currencyText = this.t('calculator.currency');
        const monthsText = this.t('calculator.months');
        
        if (currencyText && monthsText) {
            // Update currency displays safely
            const currencyElements = document.querySelectorAll('.slider-value');
            currencyElements.forEach(element => {
                if (element.innerHTML.includes('лв')) {
                    element.innerHTML = element.innerHTML.replace('лв', currencyText);
                }
                if (element.innerHTML.includes('месеца')) {
                    element.innerHTML = element.innerHTML.replace('месеца', monthsText);
                }
            });
        }
    }

    /**
     * Safely update hero section
     */
    updateHeroSafe() {
        try {
            // Hero title (contains HTML)
            const heroTitle = document.querySelector('.hero-content h1');
            const titleTranslation = this.t('hero.title');
            if (heroTitle && titleTranslation) {
                heroTitle.innerHTML = titleTranslation;
            }

            // Hero subtitle
            const heroSubtitle = document.querySelector('.hero-content p');
            const subtitleTranslation = this.t('hero.subtitle');
            if (heroSubtitle && subtitleTranslation) {
                heroSubtitle.textContent = subtitleTranslation;
            }

            // Hero stats (new simplified structure)
            const statLabels = document.querySelectorAll('.hero-stats .stat-label');
            const statKeys = ['hero.stats.approval', 'hero.stats.clients', 'hero.stats.security'];
            
            statLabels.forEach((label, index) => {
                const translation = this.t(statKeys[index]);
                if (label && translation) {
                    label.textContent = translation;
                }
            });

            // CTA button (no icon in new version)
            const ctaButton = document.querySelector('.cta-button');
            const buttonTranslation = this.t('hero.cta_button');
            if (ctaButton && buttonTranslation) {
                ctaButton.textContent = buttonTranslation;
            }
        } catch (error) {
            console.error('Error updating hero section:', error);
        }
    }

    /**
     * Safely update calculator section
     */
    updateCalculatorExpandedSafe() {
        try {
            // Calculator title and subtitle
            const calcTitle = document.querySelector('.calculator-section h2');
            const titleTranslation = this.t('calculator.title');
            if (calcTitle && titleTranslation) {
                calcTitle.textContent = titleTranslation;
            }

            const calcSubtitle = document.querySelector('.calculator-section > .container > p');
            const subtitleTranslation = this.t('calculator.subtitle');
            if (calcSubtitle && subtitleTranslation) {
                calcSubtitle.textContent = subtitleTranslation;
            }

            // Labels
            const loanAmountLabel = document.querySelector('label[for="loan-amount"]');
            const amountLabelTranslation = this.t('calculator.loan_amount');
            if (loanAmountLabel && amountLabelTranslation) {
                loanAmountLabel.textContent = amountLabelTranslation;
            }

            const loanPeriodLabel = document.querySelector('label[for="loan-period"]');
            const periodLabelTranslation = this.t('calculator.loan_period');
            if (loanPeriodLabel && periodLabelTranslation) {
                loanPeriodLabel.textContent = periodLabelTranslation;
            }

            // Result card
            const resultTitle = document.querySelector('.result-card h3');
            const resultTitleTranslation = this.t('calculator.details_title');
            if (resultTitle && resultTitleTranslation) {
                resultTitle.textContent = resultTitleTranslation;
            }

            // Result items
            const resultItems = document.querySelectorAll('.result-item span');
            const resultKeys = ['calculator.monthly_payment', 'calculator.interest_rate', 'calculator.total_amount'];
            
            resultItems.forEach((item, index) => {
                const translation = this.t(resultKeys[index]);
                if (item && translation) {
                    item.textContent = translation;
                }
            });

            // Apply button
            const applyButton = document.querySelector('.apply-button');
            const applyTranslation = this.t('calculator.apply_button');
            if (applyButton && applyTranslation) {
                applyButton.textContent = applyTranslation;
            }

            // Also update currency/months (existing safe method)
            this.updateCalculatorSafe();
        } catch (error) {
            console.error('Error updating calculator section:', error);
        }
    }

    /**
     * Safely update How It Works section
     */
    updateHowItWorksSafe() {
        try {
            const howItWorksTitle = document.querySelector('.how-it-works h2');
            const titleTranslation = this.t('how_it_works.title');
            if (howItWorksTitle && titleTranslation) {
                howItWorksTitle.textContent = titleTranslation;
            }

            const howItWorksSubtitle = document.querySelector('.how-it-works > .container > p');
            const subtitleTranslation = this.t('how_it_works.subtitle');
            if (howItWorksSubtitle && subtitleTranslation) {
                howItWorksSubtitle.textContent = subtitleTranslation;
            }

            // Steps
            const steps = document.querySelectorAll('.step .step-content h3');
            const stepDescriptions = document.querySelectorAll('.step .step-content p');
            
            steps.forEach((step, index) => {
                const stepTranslation = this.t(`how_it_works.steps.step${index + 1}.title`);
                if (step && stepTranslation) {
                    step.textContent = stepTranslation;
                }
            });

            stepDescriptions.forEach((desc, index) => {
                const descTranslation = this.t(`how_it_works.steps.step${index + 1}.description`);
                if (desc && descTranslation) {
                    desc.textContent = descTranslation;
                }
            });
        } catch (error) {
            console.error('Error updating how it works section:', error);
        }
    }

    /**
     * Safely update testimonials section
     */
    updateTestimonialsSafe() {
        try {
            // Testimonials title (new simplified structure)
            const testimonialsTitle = document.querySelector('.testimonials h2');
            const titleTranslation = this.t('testimonials.title');
            if (testimonialsTitle && titleTranslation) {
                testimonialsTitle.textContent = titleTranslation;
            }

            // Individual testimonials (new grid structure)
            const testimonialTexts = document.querySelectorAll('.testimonials-grid .testimonial p');
            const testimonialAuthors = document.querySelectorAll('.testimonials-grid .testimonial-author strong');
            const testimonialLocations = document.querySelectorAll('.testimonials-grid .testimonial-author span');

            testimonialTexts.forEach((text, index) => {
                const translation = this.t(`testimonials.testimonial${index + 1}.text`);
                if (text && translation) {
                    text.textContent = translation;
                }
            });

            testimonialAuthors.forEach((author, index) => {
                const translation = this.t(`testimonials.testimonial${index + 1}.author`);
                if (author && translation) {
                    author.textContent = translation;
                }
            });

            testimonialLocations.forEach((location, index) => {
                const translation = this.t(`testimonials.testimonial${index + 1}.location`);
                if (location && translation) {
                    location.textContent = translation;
                }
            });
        } catch (error) {
            console.error('Error updating testimonials section:', error);
        }
    }

    /**
     * Safely update success story section
     */
    updateSuccessStorySafe() {
        try {
            const successBadge = document.querySelector('.success-badge-large span');
            const badgeTranslation = this.t('success_story.badge');
            if (successBadge && badgeTranslation) {
                successBadge.textContent = badgeTranslation;
            }

            const successTitle = document.querySelector('.success-text h2');
            const titleTranslation = this.t('success_story.title');
            if (successTitle && titleTranslation) {
                successTitle.textContent = titleTranslation;
            }

            const successDescription = document.querySelector('.success-text p');
            const descTranslation = this.t('success_story.description');
            if (successDescription && descTranslation) {
                successDescription.textContent = descTranslation;
            }

            // Success highlights
            const highlightSpans = document.querySelectorAll('.highlight-item div span');
            const highlightKeys = ['success_story.highlights.housing', 'success_story.highlights.auto', 'success_story.highlights.education', 'success_story.highlights.business'];
            
            highlightSpans.forEach((span, index) => {
                const translation = this.t(highlightKeys[index]);
                if (span && translation) {
                    span.textContent = translation;
                }
            });

            // Main stat label in the stats card
            const bigLabel = document.querySelector('.main-stat .big-label');
            const bigLabelTranslation = this.t('success_story.big_stat');
            if (bigLabel && bigLabelTranslation) {
                bigLabel.textContent = bigLabelTranslation;
            }

            // Secondary stats labels
            const secondaryStatLabels = document.querySelectorAll('.secondary-stat .stat-label');
            const secondaryStatKeys = ['success_story.stats.clients', 'success_story.stats.experience', 'success_story.stats.approval'];
            
            secondaryStatLabels.forEach((label, index) => {
                const translation = this.t(secondaryStatKeys[index]);
                if (label && translation) {
                    label.textContent = translation;
                }
            });
        } catch (error) {
            console.error('Error updating success story section:', error);
        }
    }

    /**
     * Safely update FAQ section
     */
    updateFAQSafe() {
        try {
            const faqTitle = document.querySelector('.faq h2');
            const titleTranslation = this.t('faq.title');
            if (faqTitle && titleTranslation) {
                faqTitle.textContent = titleTranslation;
            }

            // FAQ items
            const faqQuestions = document.querySelectorAll('.faq-question span');
            const faqAnswers = document.querySelectorAll('.faq-answer p');

            faqQuestions.forEach((question, index) => {
                const translation = this.t(`faq.questions.q${index + 1}.question`);
                if (question && translation) {
                    question.textContent = translation;
                }
            });

            faqAnswers.forEach((answer, index) => {
                const translation = this.t(`faq.questions.q${index + 1}.answer`);
                if (answer && translation) {
                    answer.textContent = translation;
                }
            });
        } catch (error) {
            console.error('Error updating FAQ section:', error);
        }
    }

    /**
     * Safely update contact section
     */
    updateContactSafe() {
        try {
            const contactTitle = document.querySelector('.contact h2');
            const titleTranslation = this.t('contact.title');
            if (contactTitle && titleTranslation) {
                contactTitle.textContent = titleTranslation;
            }

            // Contact items
            const contactItems = document.querySelectorAll('.contact-item h3');
            const contactLabels = ['contact.phone_label', 'contact.email_label', 'contact.hours_label'];
            
            contactItems.forEach((item, index) => {
                const translation = this.t(contactLabels[index]);
                if (item && translation) {
                    item.textContent = translation;
                }
            });

            // Working hours text
            const hoursText = document.querySelector('.contact-item:last-child p');
            const hoursTranslation = this.t('contact.hours');
            if (hoursText && hoursTranslation && (hoursText.textContent.includes('Пон-Пет') || hoursText.textContent.includes('Mon-Fri'))) {
                hoursText.textContent = hoursTranslation;
            }
        } catch (error) {
            console.error('Error updating contact section:', error);
        }
    }

    /**
     * Safely update footer section
     */
    updateFooterSafe() {
        try {
            const footerDescription = document.querySelector('.footer-section p');
            const descTranslation = this.t('footer.description');
            if (footerDescription && descTranslation) {
                footerDescription.textContent = descTranslation;
            }

            const serviceTitles = document.querySelectorAll('.footer-section h4');
            if (serviceTitles.length >= 2) {
                const servicesTranslation = this.t('footer.services.title');
                const infoTranslation = this.t('footer.information.title');
                
                if (serviceTitles[0] && servicesTranslation) {
                    serviceTitles[0].textContent = servicesTranslation;
                }
                if (serviceTitles[1] && infoTranslation) {
                    serviceTitles[1].textContent = infoTranslation;
                }
            }

            // Service links
            const serviceLinks = document.querySelectorAll('.footer-section:nth-child(2) ul li a');
            const serviceKeys = ['footer.services.consumer_loans', 'footer.services.quick_loans', 'footer.services.refinancing'];
            
            serviceLinks.forEach((link, index) => {
                const translation = this.t(serviceKeys[index]);
                if (link && translation) {
                    link.textContent = translation;
                }
            });

            // Information links
            const infoLinks = document.querySelectorAll('.footer-section:nth-child(3) ul li a');
            const infoKeys = ['footer.information.terms', 'footer.information.privacy', 'footer.information.gdpr'];
            
            infoLinks.forEach((link, index) => {
                const translation = this.t(infoKeys[index]);
                if (link && translation) {
                    link.textContent = translation;
                }
            });

            // Copyright
            const copyright = document.querySelector('.footer-bottom p');
            const copyrightTranslation = this.t('footer.copyright');
            if (copyright && copyrightTranslation) {
                copyright.textContent = copyrightTranslation;
            }
        } catch (error) {
            console.error('Error updating footer section:', error);
        }
    }

    /**
     * Safely update help modal
     */
    updateHelpSafe() {
        try {
            const helpTitle = document.querySelector('.help-content h3');
            const titleTranslation = this.t('help.title');
            if (helpTitle && titleTranslation) {
                helpTitle.textContent = titleTranslation;
            }

            const helpSubtitle = document.querySelector('.help-content p');
            const subtitleTranslation = this.t('help.subtitle');
            if (helpSubtitle && subtitleTranslation) {
                helpSubtitle.textContent = subtitleTranslation;
            }

            // Help buttons
            const callButton = document.querySelector('.help-option[href^="tel"]');
            const callTranslation = this.t('help.call');
            if (callButton && callTranslation) {
                callButton.innerHTML = `<i class="fas fa-phone"></i> ${callTranslation}`;
            }

            const emailButton = document.querySelector('.help-option[href^="mailto"]');
            const emailTranslation = this.t('help.email');
            if (emailButton && emailTranslation) {
                emailButton.innerHTML = `<i class="fas fa-envelope"></i> ${emailTranslation}`;
            }

            const closeHelpButton = document.querySelector('.close-help');
            const closeTranslation = this.t('help.close');
            if (closeHelpButton && closeTranslation) {
                closeHelpButton.textContent = closeTranslation;
            }
        } catch (error) {
            console.error('Error updating help section:', error);
        }
    }

    /**
     * Safely update verification modal
     */
    updateVerificationModalSafe() {
        try {
            // Verification modal title
            const verificationTitle = document.querySelector('.verification-header h2');
            const titleTranslation = this.t('verification.title');
            if (verificationTitle && titleTranslation) {
                verificationTitle.innerHTML = `<i class="fas fa-shield-check"></i> ${titleTranslation}`;
            }

            // Step indicators
            const stepItems = document.querySelectorAll('.step-item span');
            const stepKeys = ['verification.steps.personal', 'verification.steps.documents', 'verification.steps.face'];
            
            stepItems.forEach((step, index) => {
                const translation = this.t(stepKeys[index]);
                if (step && translation) {
                    step.textContent = translation;
                }
            });

            // Step 1: Personal Information
            const personalTitle = document.querySelector('#step-1 h3');
            const personalTitleTranslation = this.t('verification.personal_info.title');
            if (personalTitle && personalTitleTranslation) {
                personalTitle.textContent = personalTitleTranslation;
            }

            // Personal info form labels
            const formLabels = [
                { selector: 'label[for="first-name"]', key: 'verification.personal_info.first_name' },
                { selector: 'label[for="last-name"]', key: 'verification.personal_info.last_name' },
                { selector: 'label[for="egn"]', key: 'verification.personal_info.egn' },
                { selector: 'label[for="phone"]', key: 'verification.personal_info.phone' },
                { selector: 'label[for="email"]', key: 'verification.personal_info.email' },
                { selector: 'label[for="address"]', key: 'verification.personal_info.address' },
                { selector: 'label[for="income"]', key: 'verification.personal_info.income' },
                { selector: 'label[for="employment"]', key: 'verification.personal_info.employment' }
            ];

            formLabels.forEach(({ selector, key }) => {
                const element = document.querySelector(selector);
                const translation = this.t(key);
                if (element && translation) {
                    element.textContent = translation;
                }
            });

            // Employment dropdown options
            const employmentSelect = document.querySelector('#employment');
            if (employmentSelect) {
                const options = employmentSelect.querySelectorAll('option');
                const optionKeys = [
                    'verification.personal_info.employment_options.select',
                    'verification.personal_info.employment_options.employed',
                    'verification.personal_info.employment_options.self_employed',
                    'verification.personal_info.employment_options.retired',
                    'verification.personal_info.employment_options.unemployed'
                ];

                options.forEach((option, index) => {
                    const translation = this.t(optionKeys[index]);
                    if (option && translation) {
                        option.textContent = translation;
                    }
                });
            }

            // Step 2: Documents
            const documentsTitle = document.querySelector('#step-2 h3');
            const docsTitleTranslation = this.t('verification.documents.title');
            if (documentsTitle && docsTitleTranslation) {
                documentsTitle.textContent = docsTitleTranslation;
            }

            // Document section titles and descriptions
            const idCardTitle = document.querySelector('#step-2 h4:first-of-type');
            const idTitleTranslation = this.t('verification.documents.id_card');
            if (idCardTitle && idTitleTranslation) {
                idCardTitle.innerHTML = `<i class="fas fa-id-card"></i> ${idTitleTranslation}`;
            }

            const idDescription = document.querySelector('#step-2 .section-description');
            const idDescTranslation = this.t('verification.documents.id_description');
            if (idDescription && idDescTranslation) {
                idDescription.textContent = idDescTranslation;
            }

            // Upload option titles
            const frontSideTitle = document.querySelector('.upload-option:first-child h5');
            const frontTranslation = this.t('verification.documents.front_side');
            if (frontSideTitle && frontTranslation) {
                frontSideTitle.textContent = frontTranslation;
            }

            const backSideTitle = document.querySelector('.upload-option:last-child h5');
            const backSideTranslation = this.t('verification.documents.back_side');
            if (backSideTitle && backSideTranslation) {
                backSideTitle.textContent = backSideTranslation;
            }

            // Upload area texts
            const uploadTexts = document.querySelectorAll('.upload-area p');
            const uploadTranslation = this.t('verification.documents.upload_text');
            uploadTexts.forEach(text => {
                if (text && uploadTranslation) {
                    text.textContent = uploadTranslation;
                }
            });

            // Camera buttons
            const cameraButtons = document.querySelectorAll('.camera-btn');
            const cameraFrontTranslation = this.t('verification.documents.camera_front');
            const cameraBackTranslation = this.t('verification.documents.camera_back');
            
            if (cameraButtons[0] && cameraFrontTranslation) {
                cameraButtons[0].innerHTML = `<i class="fas fa-camera"></i> ${cameraFrontTranslation}`;
            }
            if (cameraButtons[1] && cameraBackTranslation) {
                cameraButtons[1].innerHTML = `<i class="fas fa-camera"></i> ${cameraBackTranslation}`;
            }

            // Income document section
            const incomeTitle = document.querySelector('#step-2 h4:last-of-type');
            const incomeTitleTranslation = this.t('verification.documents.income_doc');
            if (incomeTitle && incomeTitleTranslation) {
                incomeTitle.innerHTML = `<i class="fas fa-file-invoice"></i> ${incomeTitleTranslation}`;
            }

            const incomeDescription = document.querySelector('#step-2 .section-description:last-of-type');
            const incomeDescTranslation = this.t('verification.documents.income_description');
            if (incomeDescription && incomeDescTranslation) {
                incomeDescription.textContent = incomeDescTranslation;
            }

            // Step 3: Face Verification
            const faceTitle = document.querySelector('#step-3 h3');
            const faceTitleTranslation = this.t('verification.face.title');
            if (faceTitle && faceTitleTranslation) {
                faceTitle.textContent = faceTitleTranslation;
            }

            const faceDescription = document.querySelector('#step-3 .section-description');
            const faceDescTranslation = this.t('verification.face.description');
            if (faceDescription && faceDescTranslation) {
                faceDescription.textContent = faceDescTranslation;
            }

            // Face verification tips
            const infoItems = document.querySelectorAll('.info-item span');
            const tipKeys = ['verification.face.tips.lighting', 'verification.face.tips.look', 'verification.face.tips.remove'];
            
            infoItems.forEach((item, index) => {
                const translation = this.t(tipKeys[index]);
                if (item && translation) {
                    item.textContent = translation;
                }
            });

            // Face placeholder text
            const facePlaceholder = document.querySelector('#face-placeholder p');
            const placeholderTranslation = this.t('verification.face.placeholder');
            if (facePlaceholder && placeholderTranslation) {
                facePlaceholder.textContent = placeholderTranslation;
            }

            // Face control buttons
            const startCameraBtn = document.querySelector('.camera-btn[onclick="startFaceCapture()"]');
            const startCameraTranslation = this.t('verification.face.start_camera');
            if (startCameraBtn && startCameraTranslation) {
                startCameraBtn.innerHTML = `<i class="fas fa-video"></i> ${startCameraTranslation}`;
            }

            const captureBtn = document.querySelector('.capture-btn');
            const captureTranslation = this.t('verification.face.take_photo');
            if (captureBtn && captureTranslation) {
                captureBtn.innerHTML = `<i class="fas fa-camera"></i> ${captureTranslation}`;
            }

            const uploadBtn = document.querySelector('.upload-btn');
            const uploadBtnTranslation = this.t('verification.face.upload_photo');
            if (uploadBtn && uploadBtnTranslation) {
                uploadBtn.innerHTML = `<i class="fas fa-upload"></i> ${uploadBtnTranslation}`;
            }

            // Navigation buttons
            const continueButtons = document.querySelectorAll('.next-btn');
            const backButtons = document.querySelectorAll('.prev-btn');
            const submitButton = document.querySelector('.submit-btn');

            const continueTranslation = this.t('verification.buttons.continue');
            const backButtonTranslation = this.t('verification.buttons.back');
            const submitTranslation = this.t('verification.buttons.submit');

            continueButtons.forEach(btn => {
                if (btn && continueTranslation) {
                    btn.innerHTML = `${continueTranslation} <i class="fas fa-arrow-right"></i>`;
                }
            });

            backButtons.forEach(btn => {
                if (btn && backButtonTranslation) {
                    btn.innerHTML = `<i class="fas fa-arrow-left"></i> ${backButtonTranslation}`;
                }
            });

            if (submitButton && submitTranslation) {
                submitButton.innerHTML = `${submitTranslation} <i class="fas fa-check"></i>`;
            }

            // Success step
            const successTitle = document.querySelector('#step-success h3');
            const successTitleTranslation = this.t('verification.success.title');
            if (successTitle && successTitleTranslation) {
                successTitle.textContent = successTitleTranslation;
            }

            const successMessage = document.querySelector('#step-success .success-content p');
            const successMessageTranslation = this.t('verification.success.message');
            if (successMessage && successMessageTranslation) {
                successMessage.textContent = successMessageTranslation;
            }

            const nextStepsTitle = document.querySelector('#step-success h4');
            const nextStepsTranslation = this.t('verification.success.next_steps');
            if (nextStepsTitle && nextStepsTranslation) {
                nextStepsTitle.textContent = nextStepsTranslation;
            }

            // Next steps list
            const nextStepsList = document.querySelectorAll('#step-success ol li');
            const nextStepsKeys = ['verification.success.steps.step1', 'verification.success.steps.step2', 'verification.success.steps.step3'];
            
            nextStepsList.forEach((step, index) => {
                const translation = this.t(nextStepsKeys[index]);
                if (step && translation) {
                    step.textContent = translation;
                }
            });

            const contactMessage = document.querySelector('#step-success .contact-info p:first-child');
            const contactMessageTranslation = this.t('verification.success.contact_message');
            if (contactMessage && contactMessageTranslation) {
                contactMessage.textContent = contactMessageTranslation;
            }

            const closeBtn = document.querySelector('.close-btn');
            const closeTranslation = this.t('verification.buttons.close');
            if (closeBtn && closeTranslation) {
                closeBtn.innerHTML = `${closeTranslation} <i class="fas fa-times"></i>`;
            }

        } catch (error) {
            console.error('Error updating verification modal:', error);
        }
    }

    /**
     * Safely update camera modal
     */
    updateCameraModalSafe() {
        try {
            const cameraTitle = document.querySelector('.camera-header h3');
            const titleTranslation = this.t('camera.title');
            if (cameraTitle && titleTranslation) {
                cameraTitle.textContent = titleTranslation;
            }

            const captureDocBtn = document.querySelector('.capture-document-btn');
            const captureTranslation = this.t('camera.capture');
            if (captureDocBtn && captureTranslation) {
                captureDocBtn.innerHTML = `<i class="fas fa-camera"></i> ${captureTranslation}`;
            }

            const cancelBtn = document.querySelector('.cancel-btn');
            const cancelTranslation = this.t('camera.cancel');
            if (cancelBtn && cancelTranslation) {
                cancelBtn.textContent = cancelTranslation;
            }
        } catch (error) {
            console.error('Error updating camera modal:', error);
        }
    }

    /**
     * Update any remaining small text elements and dynamic content
     */
    updateMiscellaneousSafe() {
        try {
            // Update page title if changed
            const newTitle = this.t('meta.title');
            if (newTitle && document.title !== newTitle) {
                document.title = newTitle;
            }

            // Update loading states in buttons (if any are currently showing)
            const loadingButtons = document.querySelectorAll('.loading');
            const loadingTranslation = this.t('verification.loading');
            loadingButtons.forEach(button => {
                if (button && loadingTranslation && (button.innerHTML.includes('Зареждане') || button.innerHTML.includes('Loading'))) {
                    button.innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${loadingTranslation}`;
                }
            });

            // Update any alert messages or notifications (if visible)
            // This can be expanded based on other dynamic content in your app

        } catch (error) {
            console.error('Error updating miscellaneous elements:', error);
        }
    }

    /**
     * Add observer for language changes
     */
    addObserver(callback) {
        this.observers.push(callback);
    }

    /**
     * Remove observer
     */
    removeObserver(callback) {
        const index = this.observers.indexOf(callback);
        if (index > -1) {
            this.observers.splice(index, 1);
        }
    }

    /**
     * Notify all observers of language change
     */
    notifyObservers(language) {
        this.observers.forEach(callback => {
            try {
                callback(language);
            } catch (error) {
                console.error('Error in translation observer:', error);
            }
        });
    }

    /**
     * Watch for language changes (useful for SPAs)
     */
    observeLanguageChanges() {
        // Watch for DOM changes that might need translation
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // Check if any added nodes need translation
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            const translatableElements = node.querySelectorAll('[data-translate]');
                            if (translatableElements.length > 0) {
                                this.updateTranslatedElements();
                            }
                        }
                    });
                }
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    /**
     * Get current language
     */
    getCurrentLanguage() {
        return this.currentLanguage;
    }

    /**
     * Check if language is supported
     */
    isLanguageSupported(language) {
        return language in this.translations;
    }

    /**
     * Get all supported languages
     */
    getSupportedLanguages() {
        return Object.keys(this.translations);
    }
}

// Export for use in other scripts
window.TranslationManager = TranslationManager; 