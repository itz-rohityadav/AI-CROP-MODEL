# app.py
import os
import uuid
import time
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify, session
from werkzeug.utils import secure_filename
import cv2
from PIL import Image
import logging
from datetime import datetime

# Set environment variables to avoid TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'your_crop_matters_secret_key_2023'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload44t
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Configure paths
UPLOAD_FOLDER = os.path.join('user_data', 'uploads')
LOG_FOLDER = os.path.join('logs')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs(LOG_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_FOLDER, f'app_{datetime.now().strftime("%Y%m%d")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crop_matters')

# Try to load class indices from JSON file
try:
    with open('models/class_indices.json', 'r') as f:
        class_indices = json.load(f)
        # Convert string keys to integers if needed
        if all(k.isdigit() for k in class_indices.keys()):
            class_indices = {int(k): v for k, v in class_indices.items()}
        # If the JSON has values as keys, invert the dictionary
        if all(isinstance(v, str) for v in class_indices.values()):
            DISEASE_CLASSES = list(class_indices.values())
        else:
            DISEASE_CLASSES = list(class_indices.keys())
    logger.info(f"Loaded {len(DISEASE_CLASSES)} disease classes from class_indices.json")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.warning(f"Could not load class indices: {str(e)}")
    
    DISEASE_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]
    # Disease information database
disease_database = {
    'Apple___Apple_scab': {
        'name': 'Apple Scab',
        'scientific_name': 'Venturia inaequalis',
        'description': 'A fungal disease that causes dark, scabby lesions on leaves and fruit. Severe infections can cause defoliation and reduce fruit quality and yield. Scab lesions start as olive-green spots that later become dark brown and corky.',
        'symptoms': [
            'Olive-green to brown spots on leaves',
            'Dark, corky lesions on fruit',
            'Deformed fruit',
            'Premature leaf drop',
            'Cracking of fruit skin'
        ],
        'treatment': {
            'organic': [
                'Apply sulfur or copper-based fungicides every 7-10 days during wet periods',
                'Neem oil applications at 7-14 day intervals',
                'Potassium bicarbonate sprays for light infections',
                'Bacillus subtilis biological fungicide applications'
            ],
            'chemical': [
                'Myclobutanil (Rally) applications at 7-14 day intervals',
                'Captan fungicide applications during the growing season',
                'Propiconazole sprays at recommended rates',
                'Chlorothalonil for protective treatment before infection'
            ]
        },
        'prevention': [
            'Plant resistant apple varieties like Liberty, Enterprise, or Jonafree',
            'Prune trees to improve air circulation',
            'Rake and destroy fallen leaves in autumn',
            'Apply dormant oil spray with lime sulfur during the dormant season',
            'Maintain proper tree spacing (15-20 feet apart)'
        ],
        'water_management': {
            'requirements': 'Moderate - 1 inch per week',
            'recommendations': [
                'Avoid overhead irrigation which promotes leaf wetness',
                'Water at the base of trees in the morning',
                'Ensure good soil drainage',
                'Reduce irrigation during rainy periods'
            ]
        },
        'environmental_factors': {
            'temperature': '59-75°F (15-24°C) optimal for disease development',
            'humidity': 'High humidity (>90%) promotes infection',
            'conditions': 'Wet spring weather significantly increases risk'
        },
        'crop_impact': {
            'yield_reduction': '50-70% in severe cases',
            'quality_impact': 'Significant reduction in fruit marketability',
            'economic_threshold': 'Treatment justified when 1-5% of leaves show symptoms'
        }
    },
    
    'Apple___Black_rot': {
        'name': 'Apple Black Rot',
        'scientific_name': 'Botryosphaeria obtusa',
        'description': 'A fungal disease affecting apple trees that causes leaf spots, fruit rot, and cankers on branches. The disease can overwinter in cankers and mummified fruit, spreading during warm, wet conditions.',
        'symptoms': [
            'Purple spots on leaves that develop tan or brown centers',
            'Dark, sunken lesions on fruit that eventually turn black',
            'Mummified fruit that remain attached to the tree',
            'Cankers on branches and twigs',
            'Leaf yellowing and premature drop'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied at 7-10 day intervals',
                'Neem oil applications during the growing season',
                'Potassium bicarbonate sprays for mild infections',
                'Bacillus subtilis biological treatments'
            ],
            'chemical': [
                'Thiophanate-methyl fungicides during the growing season',
                'Captan applications at 10-14 day intervals',
                'Myclobutanil treatments from petal fall through harvest',
                'Mancozeb sprays during early season development'
            ]
        },
        'prevention': [
            'Remove and destroy infected plant material including mummified fruit',
            'Prune out cankers and dead wood during dormant season',
            'Maintain tree vigor through proper fertilization',
            'Ensure adequate spacing between trees for air circulation',
            'Apply dormant oil sprays before bud break'
        ],
        'water_management': {
            'requirements': 'Moderate - 1 inch per week',
            'recommendations': [
                'Avoid overhead irrigation to reduce leaf wetness',
                'Water at the base of trees in early morning',
                'Ensure proper soil drainage',
                'Adjust irrigation during rainy periods'
            ]
        },
        'environmental_factors': {
            'temperature': '68-75°F (20-24°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Warm, wet weather accelerates disease spread'
        },
        'crop_impact': {
            'yield_reduction': '30-50% in severe cases',
            'quality_impact': 'Significant reduction in marketable fruit',
            'economic_threshold': 'Treatment warranted when 3-5% of fruit show symptoms'
        }
    },
        'Apple___Cedar_apple_rust': {
        'name': 'Cedar Apple Rust',
        'scientific_name': 'Gymnosporangium juniperi-virginianae',
        'description': 'A fungal disease requiring both apple trees and cedar/juniper trees to complete its life cycle. It causes bright orange-yellow spots on apple leaves and fruit, and gelatinous orange telial horns on cedar trees.',
        'symptoms': [
            'Bright orange-yellow spots on apple leaves and fruit',
            'Small, raised bumps on the undersides of infected leaves',
            'Defoliation in severe cases',
            'Distorted fruit with yellow-orange lesions',
            'Reduced fruit quality and yield'
        ],
        'treatment': {
            'organic': [
                'Sulfur-based fungicides applied preventatively',
                'Neem oil applications every 7-10 days during spring',
                'Potassium bicarbonate sprays for light infections',
                'Bacillus subtilis biological fungicide treatments'
            ],
            'chemical': [
                'Myclobutanil (Rally) applications starting at pink bud stage',
                'Propiconazole sprays at 10-14 day intervals',
                'Mancozeb treatments during primary infection periods',
                'Fenarimol applications from green tip through petal fall'
            ]
        },
        'prevention': [
            'Remove nearby cedar/juniper trees if possible (within 1-2 miles)',
            'Plant resistant apple varieties like Liberty, Redfree, or Williams Pride',
            'Apply preventative fungicides before infection periods',
            'Maintain good air circulation through proper pruning',
            'Destroy fallen infected leaves'
        ],
        'water_management': {
            'requirements': 'Moderate - 1 inch per week',
            'recommendations': [
                'Avoid overhead irrigation which promotes spore germination',
                'Water at the base of trees in the morning',
                'Ensure good soil drainage',
                'Reduce irrigation during high humidity periods'
            ]
        },
        'environmental_factors': {
            'temperature': '50-75°F (10-24°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Spring rains and moderate temperatures increase risk'
        },
        'crop_impact': {
            'yield_reduction': '20-40% in severe cases',
            'quality_impact': 'Reduced marketability of spotted fruit',
            'economic_threshold': 'Treatment justified when disease appears on 5% of leaves'
        }
    },
    
    'Apple___healthy': {
        'name': 'Healthy Apple',
        'scientific_name': 'Malus domestica',
        'description': 'Healthy apple trees display vibrant green leaves, sturdy branches, and produce firm, well-colored fruit. Proper maintenance ensures continued tree health and productivity.',
        'symptoms': [
            'Vibrant green leaves without spots or discoloration',
            'Uniform leaf size and shape',
            'Firm, well-developed fruit',
            'Strong, flexible branches',
            'Consistent annual growth'
        ],
        'treatment': {
            'organic': [
                'Regular applications of compost tea for soil health',
                'Foliar seaweed sprays to boost micronutrients',
                'Beneficial insect releases for pest management',
                'Organic mulch application around tree base'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications as needed',
                'Preventative fungicide program during wet seasons',
                'Dormant oil sprays during winter',
                'Calcium sprays to prevent bitter pit in fruit'
            ]
        },
        'prevention': [
            'Regular pruning to maintain tree structure and air circulation',
            'Annual soil testing and appropriate amendments',
            'Proper tree spacing (15-20 feet apart)',
            'Mulching to retain moisture and suppress weeds',
            'Integrated pest management monitoring'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-1.5 inches per week during growing season',
            'recommendations': [
                'Deep watering to encourage deep root development',
                'Drip irrigation to deliver water efficiently',
                'Morning watering to reduce evaporation',
                'Adjust watering based on rainfall and temperature'
            ]
        },
        'environmental_factors': {
            'temperature': '60-75°F (15-24°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best fruit production'
        },
        'crop_impact': {
            'yield_potential': '100-300 pounds per mature tree annually',
            'quality_characteristics': 'Firm, well-colored, properly sized fruit',
            'economic_value': 'Maximum marketability and storage potential'
        }
    },
    
    'Blueberry___healthy': {
        'name': 'Healthy Blueberry',
        'scientific_name': 'Vaccinium corymbosum',
        'description': 'Healthy blueberry plants have vibrant green leaves, strong canes, and produce plump, dusty-blue berries. They require acidic soil and consistent moisture for optimal growth and production.',
        'symptoms': [
            'Bright green leaves with no spots or discoloration',
            'Firm, upright canes with no lesions',
            'Abundant white or pink bell-shaped flowers in spring',
            'Plump, dusty-blue berries with a waxy bloom',
            'Consistent annual growth of new canes'
        ],
        'treatment': {
            'organic': [
                'Application of acidic organic matter like pine needles or coffee grounds',
                'Elemental sulfur to maintain soil pH',
                'Organic fish emulsion fertilizer during growing season',
                'Compost tea applications to boost soil biology'
            ],
            'chemical': [
                'Balanced acidic fertilizer (specifically for acid-loving plants)',
                'Chelated iron supplements for iron chlorosis prevention',
                'Ammonium sulfate applications to maintain acidity',
                'Micronutrient sprays as indicated by soil tests'
            ]
        },
        'prevention': [
            'Maintain soil pH between 4.5-5.5',
            'Mulch with pine bark, sawdust, or pine needles',
            'Proper plant spacing (4-6 feet apart)',
            'Regular pruning to remove old canes and maintain vigor',
            'Bird netting to protect ripening fruit'
        ],
        'water_management': {
            'requirements': 'Consistent moisture - 1-2 inches per week',
            'recommendations': [
                'Drip irrigation to provide consistent moisture',
                'Avoid overhead watering which can promote disease',
                'Ensure good drainage as plants dislike wet feet',
                'Increase watering during fruit development and hot periods'
            ]
        },
        'environmental_factors': {
            'temperature': '60-75°F (15-24°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun to partial shade, with protection from harsh afternoon sun in hot climates'
        },
        'crop_impact': {
            'yield_potential': '5-10 pounds per mature bush annually',
            'quality_characteristics': 'Firm, sweet berries with good size and bloom',
            'economic_value': 'High marketability for fresh consumption and processing'
        }
    },
    
    'Cherry_(including_sour)___Powdery_mildew': {
        'name': 'Cherry Powdery Mildew',
        'scientific_name': 'Podosphaera clandestina',
        'description': 'A fungal disease affecting cherry trees that appears as a white powdery coating on leaves, shoots, and fruit. Severe infections can reduce photosynthesis, stunt growth, and decrease fruit quality and yield.',
        'symptoms': [
            'White powdery coating on leaves, especially the undersides',
            'Curling, twisting, or stunting of new leaves and shoots',
            'Reduced fruit size and quality',
            'Premature leaf drop in severe cases',
            'Russeting or scarring of fruit'
        ],
        'treatment': {
            'organic': [
                'Sulfur applications at 7-10 day intervals',
                'Potassium bicarbonate sprays for active infections',
                'Neem oil treatments every 7-14 days',
                'Milk spray (1:10 milk to water ratio) as a preventative'
            ],
            'chemical': [
                'Myclobutanil applications at first sign of disease',
                'Trifloxystrobin sprays at 14-day intervals',
                'Propiconazole treatments during growing season',
                'Tebuconazole applications for severe infections'
            ]
        },
        'prevention': [
            'Plant resistant cherry varieties when available',
            'Prune trees to improve air circulation',
            'Avoid excessive nitrogen fertilization',
            'Remove and destroy infected plant parts',
            'Space trees properly to reduce humidity in canopy'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-1.5 inches per week',
            'recommendations': [
                'Avoid overhead irrigation which increases humidity',
                'Water at the base of trees in the morning',
                'Ensure good soil drainage',
                'Avoid overwatering which can increase humidity around plants'
            ]
        },
        'environmental_factors': {
            'temperature': '60-80°F (15-27°C) optimal for disease development',
            'humidity': 'High humidity promotes infection, but free water inhibits spore germination',
            'conditions': 'Shade and poor air circulation increase disease severity'
        },
        'crop_impact': {
            'yield_reduction': '20-30% in severe cases',
            'quality_impact': 'Reduced marketability due to fruit scarring',
            'economic_threshold': 'Treatment warranted when disease appears on 3-5% of leaves'
        }
    },
        'Cherry_(including_sour)___healthy': {
        'name': 'Healthy Cherry',
        'scientific_name': 'Prunus avium (sweet) / Prunus cerasus (sour)',
        'description': 'Healthy cherry trees display glossy, dark green leaves, strong branches, and produce firm, brightly colored fruit. Proper maintenance ensures tree health, longevity, and optimal fruit production.',
        'symptoms': [
            'Glossy, dark green leaves without spots or discoloration',
            'Uniform leaf size and shape',
            'Firm, well-developed fruit with bright coloration',
            'Strong, flexible branches with smooth bark',
            'Consistent annual growth and flowering'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications in spring',
                'Compost tea applications to boost soil biology',
                'Seaweed extract foliar sprays for micronutrients',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Preventative fungicide program during wet seasons',
                'Dormant oil sprays during winter dormancy',
                'Calcium sprays to improve fruit firmness'
            ]
        },
        'prevention': [
            'Regular pruning to maintain open canopy and air circulation',
            'Annual soil testing and appropriate amendments',
            'Proper tree spacing (15-20 feet for sweet, 12-15 feet for sour)',
            'Mulching to retain moisture and suppress weeds',
            'Integrated pest management monitoring'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-1.5 inches per week during growing season',
            'recommendations': [
                'Deep watering to encourage deep root development',
                'Drip irrigation to deliver water efficiently',
                'Morning watering to reduce evaporation',
                'Adjust watering based on rainfall and temperature'
            ]
        },
        'environmental_factors': {
            'temperature': '60-75°F (15-24°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best fruit production'
        },
        'crop_impact': {
            'yield_potential': '30-50 pounds per mature sweet cherry tree; 15-20 pounds for sour cherry',
            'quality_characteristics': 'Firm, well-colored, properly sized fruit with good sugar content',
            'economic_value': 'Maximum marketability and storage potential'
        }
    },
    
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'name': 'Gray Leaf Spot',
        'scientific_name': 'Cercospora zeae-maydis',
        'description': 'A fungal disease of corn that causes rectangular, grayish-brown lesions on leaves. Severe infections can lead to significant yield losses by reducing photosynthetic area and weakening plants.',
        'symptoms': [
            'Rectangular, grayish-brown to tan lesions parallel to leaf veins',
            'Lesions with a distinct yellow halo in early stages',
            'Lesions that expand and coalesce under favorable conditions',
            'Lower leaves affected first with upward progression',
            'Premature death of leaves in severe cases'
        ],
        'treatment': {
            'organic': [
                'Limited effective organic treatments available',
                'Copper-based fungicides may provide some suppression',
                'Biological fungicides containing Bacillus subtilis',
                'Crop rotation to non-host crops for at least one year'
            ],
            'chemical': [
                'Strobilurin (QoI) fungicides like azoxystrobin or pyraclostrobin',
                'Triazole fungicides such as propiconazole or tebuconazole',
                'Combination products containing both QoI and triazole fungicides',
                'Timing applications at early disease detection through tasseling'
            ]
        },
        'prevention': [
            'Plant resistant hybrids with high gray leaf spot tolerance',
            'Implement crop rotation with non-host crops for 1-2 years',
            'Practice conservation tillage to bury infected residue',
            'Avoid excessive nitrogen applications',
            'Plant in well-drained fields with good air circulation'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Avoid overhead irrigation which promotes leaf wetness',
                'If irrigating, do so in early morning to allow leaves to dry',
                'Ensure proper field drainage',
                'Manage irrigation to avoid drought stress which can weaken plants'
            ]
        },
        'environmental_factors': {
            'temperature': '75-85°F (24-29°C) optimal for disease development',
            'humidity': 'High humidity (>90%) and leaf wetness for 12+ hours promotes infection',
            'conditions': 'Disease severity increases with extended periods of warm, humid weather'
        },
        'crop_impact': {
            'yield_reduction': '20-50% in severe cases',
            'quality_impact': 'Reduced grain fill and lower test weights',
            'economic_threshold': 'Treatment warranted when lesions appear on lower third of plant before tasseling'
        }
    },
    
    'Corn_(maize)___Common_rust_': {
        'name': 'Common Rust of Corn',
        'scientific_name': 'Puccinia sorghi',
        'description': 'A fungal disease characterized by small, circular to elongated, brick-red to brown pustules on corn leaves. The disease can reduce photosynthetic area and yield in severe cases, particularly in sweet corn.',
        'symptoms': [
            'Small, circular to elongated brick-red to brown pustules on both leaf surfaces',
            'Pustules that rupture to release reddish-brown spores',
            'Pustules turning black as they age',
            'Chlorotic (yellow) areas surrounding pustules',
            'Severe infections causing leaf death and reduced plant vigor'
        ],
        'treatment': {
            'organic': [
                'Limited effective organic treatments available',
                'Sulfur-based fungicides may provide some protection',
                'Copper-based fungicides as preventative treatment',
                'Biological fungicides containing Bacillus subtilis'
            ],
            'chemical': [
                'Strobilurin (QoI) fungicides like azoxystrobin or pyraclostrobin',
                'Triazole fungicides such as propiconazole or tebuconazole',
                'Chlorothalonil as a protective fungicide',
                'Mancozeb applications at 7-14 day intervals'
            ]
        },
        'prevention': [
            'Plant resistant hybrids with rust tolerance',
            'Early planting to avoid peak rust spore periods',
            'Proper plant spacing to improve air circulation',
            'Balanced fertilization to promote plant health',
            'Scout fields regularly for early detection'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Avoid overhead irrigation which promotes spore germination',
                'If irrigating, do so in early morning to allow leaves to dry quickly',
                'Ensure proper field drainage',
                'Manage irrigation to avoid plant stress'
            ]
        },
        'environmental_factors': {
            'temperature': '60-77°F (16-25°C) optimal for disease development',
            'humidity': 'High humidity and 6+ hours of leaf wetness promotes infection',
            'conditions': 'Cool, moist conditions favor disease development'
        },
        'crop_impact': {
            'yield_reduction': '10-30% in severe cases, higher in sweet corn',
            'quality_impact': 'Reduced ear size and kernel fill',
            'economic_threshold': 'Treatment warranted in sweet corn when rust appears before silking; field corn typically has higher tolerance'
        }
    },
    
    'Corn_(maize)___Northern_Leaf_Blight': {
        'name': 'Northern Corn Leaf Blight',
        'scientific_name': 'Exserohilum turcicum (formerly Helminthosporium turcicum)',
        'description': 'A fungal disease characterized by large, cigar-shaped lesions on corn leaves. Severe infections can cause significant yield losses by reducing photosynthetic area and weakening plants.',
        'symptoms': [
            'Large, cigar-shaped gray-green to tan lesions (1-6 inches long)',
            'Lesions that develop dark areas of fungal sporulation',
            'Lower leaves affected first with upward progression',
            'Lesions that may coalesce to blight entire leaves',
            'Premature death of leaves in severe cases'
        ],
        'treatment': {
            'organic': [
                'Limited effective organic treatments available',
                'Copper-based fungicides may provide some suppression',
                'Biological fungicides containing Bacillus subtilis',
                'Crop rotation to non-host crops for at least one year'
            ],
            'chemical': [
                'Strobilurin (QoI) fungicides like azoxystrobin or pyraclostrobin',
                'Triazole fungicides such as propiconazole or tebuconazole',
                'Combination products containing both QoI and triazole fungicides',
                'Timing applications at early disease detection through silking'
            ]
        },
        'prevention': [
            'Plant resistant hybrids with Ht genes for NCLB resistance',
            'Implement crop rotation with non-host crops for 1-2 years',
            'Practice conservation tillage to bury infected residue',
            'Balanced fertilization to promote plant health',
            'Plant in well-drained fields with good air circulation'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Avoid overhead irrigation which promotes leaf wetness',
                'If irrigating, do so in early morning to allow leaves to dry',
                'Ensure proper field drainage',
                'Manage irrigation to avoid drought stress which can weaken plants'
            ]
        },
        'environmental_factors': {
            'temperature': '65-80°F (18-27°C) optimal for disease development',
            'humidity': 'High humidity and 6+ hours of leaf wetness promotes infection',
            'conditions': 'Disease severity increases with extended periods of wet, moderate temperature weather'
        },
        'crop_impact': {
            'yield_reduction': '30-50% in severe cases',
            'quality_impact': 'Reduced grain fill and lower test weights',
            'economic_threshold': 'Treatment warranted when lesions appear on lower third of plant before tasseling'
        }
    },
        'Corn_(maize)___healthy': {
        'name': 'Healthy Corn',
        'scientific_name': 'Zea mays',
        'description': 'Healthy corn plants display vibrant green leaves, strong stalks, and well-developed ears. Proper maintenance ensures optimal growth, yield, and resistance to pests and diseases.',
        'symptoms': [
            'Uniform, vibrant green leaves without lesions or discoloration',
            'Strong, upright stalks without lodging',
            'Well-formed tassels with abundant pollen',
            'Properly filled ears with plump kernels',
            'Consistent growth rate and development'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications based on soil tests',
                'Compost or manure incorporation before planting',
                'Foliar seaweed or fish emulsion sprays during growth',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Split nitrogen applications for optimal utilization',
                'Micronutrient supplements as indicated by tissue tests',
                'Preventative fungicide program in high-risk situations'
            ]
        },
        'prevention': [
            'Crop rotation with non-grass crops',
            'Proper hybrid selection for local conditions',
            'Optimal planting date for your region',
            'Appropriate plant population (28,000-34,000 plants/acre)',
            'Regular scouting for early pest and disease detection'
        ],
        'water_management': {
            'requirements': 'Moderate to high - 1-2 inches per week, critical during silking and grain fill',
            'recommendations': [
                'Ensure adequate moisture during critical growth stages',
                'Monitor soil moisture to avoid stress',
                'Implement efficient irrigation systems if available',
                'Maintain field drainage to prevent waterlogging'
            ]
        },
        'environmental_factors': {
            'temperature': '77-91°F (25-33°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun for maximum photosynthesis and yield'
        },
        'crop_impact': {
            'yield_potential': '150-250 bushels per acre under optimal conditions',
            'quality_characteristics': 'High test weight, proper moisture content, minimal damage',
            'economic_value': 'Maximum marketability and feeding value'
        }
    },
    
    'Grape___Black_rot': {
        'name': 'Grape Black Rot',
        'scientific_name': 'Guignardia bidwellii',
        'description': 'A fungal disease affecting grapes that causes leaf spots, fruit rot, and stem lesions. Black rot is one of the most serious diseases of grapes in regions with warm, humid growing seasons.',
        'symptoms': [
            'Circular, tan to brown spots with dark borders on leaves',
            'Small, black fruiting bodies (pycnidia) visible in leaf lesions',
            'Tan to brown lesions on shoots and stems',
            'Fruit initially showing small white dots, then brown rot',
            'Infected berries shrivel into hard, black mummies that may remain attached to clusters'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively',
                'Potassium bicarbonate sprays for light infections',
                'Sulfur applications at 7-14 day intervals',
                'Bacillus subtilis biological fungicide treatments'
            ],
            'chemical': [
                'Mancozeb applications starting at early shoot growth',
                'Myclobutanil treatments at 10-14 day intervals',
                'Tebuconazole sprays from pre-bloom through veraison',
                'Azoxystrobin applications during critical infection periods'
            ]
        },
        'prevention': [
            'Remove all mummified berries and infected canes during dormant pruning',
            'Maintain open canopy through proper pruning for air circulation',
            'Remove infected leaves and fruit during the growing season',
            'Plant resistant varieties when possible',
            'Apply preventative fungicides before rainfall events'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water in the morning to allow foliage to dry quickly',
                'Avoid overhead irrigation which promotes disease',
                'Ensure good soil drainage to prevent root issues'
            ]
        },
        'environmental_factors': {
            'temperature': '70-85°F (21-29°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Rainfall or extended periods of leaf wetness (6+ hours) significantly increases infection risk'
        },
        'crop_impact': {
            'yield_reduction': '50-80% in severe cases',
            'quality_impact': 'Complete loss of affected fruit',
            'economic_threshold': 'Treatment warranted at first sign of disease'
        }
    },
    
    'Grape___Esca_(Black_Measles)': {
        'name': 'Esca (Black Measles)',
        'scientific_name': 'Complex of fungi including Phaeomoniella chlamydospora, Phaeoacremonium spp., and Fomitiporia mediterranea',
        'description': 'A complex fungal disease affecting the woody tissue of grapevines, causing leaf discoloration, fruit spotting, and eventual vine decline. Also known as black measles or apoplexy when symptoms develop rapidly.',
        'symptoms': [
            'Interveinal chlorosis and necrosis on leaves, creating a tiger-stripe pattern',
            'Small, dark purple to black spots on fruit',
            'Stunted shoot growth and dieback',
            'Cross-sectional wood discoloration and decay',
            'Sudden wilting and death of entire vines (apoplexy) in hot weather'
        ],
        'treatment': {
            'organic': [
                'Limited effective treatments once established',
                'Wound protectants containing boric acid after pruning',
                'Trichoderma-based biological products for pruning wounds',
                'Trunk renewal by training new shoots from the base'
            ],
            'chemical': [
                'No chemical treatments cure established infections',
                'Fungicide wound protectants after pruning',
                'Thiophanate-methyl applications to protect pruning wounds',
                'Preventative fungicides during early vine establishment'
            ]
        },
        'prevention': [
            'Use clean, certified planting material',
            'Prune during dry weather to reduce infection risk',
            'Apply wound protectants immediately after pruning',
            'Remove and destroy infected wood and dead vines',
            'Avoid pruning stress by maintaining balanced crop loads'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Avoid water stress which can trigger apoplexy symptoms',
                'Maintain consistent soil moisture',
                'Use drip irrigation for efficient water delivery',
                'Ensure good soil drainage to prevent root stress'
            ]
        },
        'environmental_factors': {
            'temperature': 'High temperatures (>90°F/32°C) can trigger apoplexy in infected vines',
            'humidity': 'Wet conditions during pruning increase infection risk',
            'conditions': 'Stress factors like drought or excessive crop can worsen symptoms'
        },
        'crop_impact': {
            'yield_reduction': '10-50% depending on severity',
            'quality_impact': 'Reduced fruit quality and marketability',
            'economic_threshold': 'Progressive disease requiring preventative management; no threshold for treatment'
        }
    },
    
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'name': 'Grape Leaf Blight',
        'scientific_name': 'Pseudocercospora vitis (formerly Isariopsis leaf spot)',
        'description': 'A fungal disease affecting grape leaves, causing spots that can coalesce into blighted areas. Severe infections can cause premature defoliation, reducing vine vigor and fruit quality.',
        'symptoms': [
            'Small, irregular dark brown to black spots on leaves',
            'Spots that enlarge and develop gray-brown centers with dark borders',
            'Lesions that may coalesce to form larger blighted areas',
            'Premature leaf drop in severe cases',
            'Reduced fruit ripening due to defoliation'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively',
                'Sulfur applications at 7-14 day intervals',
                'Potassium bicarbonate sprays for light infections',
                'Bacillus subtilis biological fungicide treatments'
            ],
            'chemical': [
                'Mancozeb applications starting at early shoot growth',
                'Azoxystrobin treatments at 10-14 day intervals',
                'Tebuconazole sprays during critical infection periods',
                'Myclobutanil applications from pre-bloom through veraison'
            ]
        },
        'prevention': [
            'Maintain open canopy through proper pruning for air circulation',
            'Remove infected leaves during the growing season',
            'Apply preventative fungicides before rainfall events',
            'Ensure balanced nutrition to promote vine health',
            'Avoid overhead irrigation which promotes leaf wetness'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water in the morning to allow foliage to dry quickly',
                'Avoid overhead irrigation which promotes disease',
                'Ensure good soil drainage to prevent root issues'
            ]
        },
        'environmental_factors': {
            'temperature': '68-82°F (20-28°C) optimal for disease development',
            'humidity': 'High humidity (>85%) promotes infection',
            'conditions': 'Prolonged leaf wetness (6+ hours) significantly increases infection risk'
        },
        'crop_impact': {
            'yield_reduction': '10-30% in severe cases',
            'quality_impact': 'Reduced fruit quality due to poor ripening',
            'economic_threshold': 'Treatment warranted when disease appears on 3-5% of leaves'
        }
    },
        'Grape___healthy': {
        'name': 'Healthy Grape',
        'scientific_name': 'Vitis vinifera / Vitis labrusca / Vitis hybrid',
        'description': 'Healthy grapevines display vibrant green leaves, strong canes, and well-developed fruit clusters. Proper maintenance ensures vine longevity, optimal yield, and high-quality fruit production.',
        'symptoms': [
            'Vibrant green leaves without spots or discoloration',
            'Uniform leaf size and shape appropriate to variety',
            'Strong, flexible canes with appropriate growth',
            'Well-formed fruit clusters with uniform berry development',
            'Consistent annual growth and fruit production'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications based on soil tests',
                'Compost applications to improve soil health',
                'Seaweed extract foliar sprays for micronutrients',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Preventative fungicide program during critical periods',
                'Micronutrient supplements as indicated by tissue tests',
                'Growth regulators when needed for cluster development'
            ]
        },
        'prevention': [
            'Regular pruning to maintain open canopy and air circulation',
            'Shoot thinning and positioning for optimal sun exposure',
            'Leaf removal around fruit clusters when appropriate',
            'Cover crops for soil health and beneficial insect habitat',
            'Regular monitoring for early pest and disease detection'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Drip irrigation for efficient water delivery',
                'Regulated deficit irrigation during specific growth stages',
                'Soil moisture monitoring to optimize irrigation timing',
                'Adjust watering based on growth stage and weather conditions'
            ]
        },
        'environmental_factors': {
            'temperature': '70-85°F (21-29°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best fruit development'
        },
        'crop_impact': {
            'yield_potential': '3-5 tons per acre for wine grapes; 5-10 tons for table grapes',
            'quality_characteristics': 'Proper sugar/acid balance, varietal flavor development, uniform ripening',
            'economic_value': 'Maximum marketability and wine quality potential'
        }
    },
    
    'Orange___Haunglongbing_(Citrus_greening)': {
        'name': 'Citrus Greening (Huanglongbing)',
        'scientific_name': 'Candidatus Liberibacter asiaticus',
        'description': 'A devastating bacterial disease spread by the Asian citrus psyllid that affects all citrus varieties. It causes mottled leaves, misshapen and bitter fruit, and eventual tree decline and death.',
        'symptoms': [
            'Asymmetrical blotchy mottling of leaves',
            'Yellow shoots in an otherwise green canopy',
            'Misshapen, small fruit with uneven coloration',
            'Bitter, inedible fruit with aborted seeds',
            'Premature fruit drop and overall tree decline'
        ],
        'treatment': {
            'organic': [
                'No effective organic treatments for the bacteria',
                'Psyllid management through organic insecticides like neem oil',
                'Beneficial insect releases for psyllid control',
                'Enhanced nutrition programs to extend productive tree life'
            ],
            'chemical': [
                'No curative treatments available',
                'Systemic insecticides for psyllid control',
                'Foliar insecticides for adult psyllid management',
                'Enhanced micronutrient applications to maintain tree health'
            ]
        },
        'prevention': [
            'Plant certified disease-free trees from reputable nurseries',
            'Regular monitoring and control of Asian citrus psyllid',
            'Remove and destroy infected trees to prevent spread',
            'Coordinate area-wide psyllid management programs',
            'Avoid moving plant material from infected areas'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week depending on climate',
            'recommendations': [
                'Maintain consistent soil moisture to reduce tree stress',
                'Avoid drought stress which can worsen symptoms',
                'Use efficient irrigation systems like micro-sprinklers',
                'Adjust irrigation based on tree health and weather conditions'
            ]
        },
        'environmental_factors': {
            'temperature': 'Disease progression faster in warm temperatures (75-85°F/24-29°C)',
            'humidity': 'Psyllid populations higher in humid conditions',
            'conditions': 'Year-round pressure in tropical/subtropical climates'
        },
        'crop_impact': {
            'yield_reduction': '30-100% depending on infection stage',
            'quality_impact': 'Fruit becomes unmarketable due to poor size, appearance, and taste',
            'economic_threshold': 'Zero tolerance - management required at first detection'
        }
    },
    
    'Peach___Bacterial_spot': {
        'name': 'Peach Bacterial Spot',
        'scientific_name': 'Xanthomonas arboricola pv. pruni',
        'description': 'A bacterial disease affecting peach trees that causes spots on leaves, fruit, and twigs. Severe infections can cause defoliation, reduced fruit quality, and increased susceptibility to other stresses.',
        'symptoms': [
            'Small, water-soaked lesions on leaves that become angular and purple-brown',
            'Shot-hole appearance as leaf lesions fall out',
            'Circular, sunken lesions on fruit that may crack',
            'Sunken, dark cankers on twigs and branches',
            'Premature leaf drop in severe cases'
        ],
        'treatment': {
            'organic': [
                'Copper-based bactericides applied preventatively',
                'Reduce application rates during active growth to prevent phytotoxicity',
                'Bacillus subtilis biological treatments',
                'Hydrogen dioxide products for surface sterilization'
            ],
            'chemical': [
                'Copper-based bactericides applied at leaf fall and during dormancy',
                'Oxytetracycline applications during growing season',
                'Copper-mancozeb combinations for better efficacy',
                'Chemical thinning to reduce fruit-to-fruit contact'
            ]
        },
        'prevention': [
            'Plant resistant varieties like Redhaven, Sunhigh, or Candor',
            'Prune during dry weather to promote air circulation',
            'Remove and destroy infected twigs during dormant season',
            'Avoid overhead irrigation which promotes bacterial spread',
            'Maintain balanced nutrition, avoiding excess nitrogen'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water at the base of trees in the morning',
                'Ensure good soil drainage',
                'Avoid irrigation during high humidity periods'
            ]
        },
        'environmental_factors': {
            'temperature': '75-85°F (24-29°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Rainfall, overhead irrigation, and wind-driven rain increase spread'
        },
        'crop_impact': {
            'yield_reduction': '10-30% in severe cases',
            'quality_impact': 'Reduced marketability due to fruit blemishes',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of fruit or foliage'
        }
    },
    
    'Peach___healthy': {
        'name': 'Healthy Peach',
        'scientific_name': 'Prunus persica',
        'description': 'Healthy peach trees display vibrant green leaves, strong branches, and produce firm, well-colored fruit. Proper maintenance ensures tree health, productivity, and fruit quality.',
        'symptoms': [
            'Vibrant green leaves without spots or discoloration',
            'Uniform leaf size and shape',
            'Firm, well-developed fruit with appropriate blush',
            'Strong, flexible branches with smooth bark',
            'Consistent annual growth and flowering'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications in spring',
                'Compost applications to improve soil health',
                'Seaweed extract foliar sprays for micronutrients',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Preventative fungicide program during critical periods',
                'Dormant oil sprays during winter',
                'Growth regulators when needed for fruit thinning'
            ]
        },
        'prevention': [
            'Regular pruning to maintain open center and air circulation',
            'Annual dormant pruning to remove dead wood and shape tree',
            'Proper fruit thinning to improve size and quality',
            'Mulching to retain moisture and suppress weeds',
            'Regular monitoring for early pest and disease detection'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Deep watering to encourage deep root development',
                'Drip irrigation to deliver water efficiently',
                'Morning watering to reduce evaporation',
                'Adjust watering based on fruit development stage'
            ]
        },
        'environmental_factors': {
            'temperature': '65-80°F (18-27°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best fruit development'
        },
        'crop_impact': {
            'yield_potential': '100-150 pounds per mature tree annually',
            'quality_characteristics': 'Firm, well-colored fruit with good sugar/acid balance',
            'economic_value': 'Maximum marketability and storage potential'
        }
    },
        'Pepper,_bell___Bacterial_spot': {
        'name': 'Pepper Bacterial Spot',
        'scientific_name': 'Xanthomonas campestris pv. vesicatoria',
        'description': 'A bacterial disease affecting bell peppers that causes spots on leaves, stems, and fruit. Severe infections can lead to defoliation, sunscald of fruit, and significant yield losses.',
        'symptoms': [
            'Small, water-soaked lesions on leaves that become brown with yellow halos',
            'Raised, scabby spots on fruit that start as small blisters',
            'Defoliation in severe cases exposing fruit to sunscald',
            'Dark, elongated lesions on stems and petioles',
            'Premature fruit drop in severe infections'
        ],
        'treatment': {
            'organic': [
                'Copper-based bactericides applied preventatively',
                'Bacillus subtilis biological treatments',
                'Hydrogen dioxide products for surface sterilization',
                'Crop rotation with non-solanaceous crops for 2-3 years'
            ],
            'chemical': [
                'Copper-based bactericides applied at 7-10 day intervals',
                'Copper-mancozeb combinations for better efficacy',
                'Acibenzolar-S-methyl (Actigard) as a plant activator',
                'Streptomycin applications in seedling production (where legal)'
            ]
        },
        'prevention': [
            'Plant certified disease-free seeds and transplants',
            'Use resistant varieties when available',
            'Avoid overhead irrigation which promotes bacterial spread',
            'Practice crop rotation with non-solanaceous crops',
            'Remove and destroy infected plant debris after harvest'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water at the base of plants in the morning',
                'Ensure good soil drainage',
                'Avoid working with plants when foliage is wet'
            ]
        },
        'environmental_factors': {
            'temperature': '75-86°F (24-30°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Rainfall, overhead irrigation, and wind-driven rain increase spread'
        },
        'crop_impact': {
            'yield_reduction': '10-50% in severe cases',
            'quality_impact': 'Reduced marketability due to fruit blemishes',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of plants'
        }
    },
    
    'Pepper,_bell___healthy': {
        'name': 'Healthy Bell Pepper',
        'scientific_name': 'Capsicum annuum',
        'description': 'Healthy bell pepper plants display dark green leaves, strong stems, and produce firm, well-colored fruit. Proper maintenance ensures plant vigor, productivity, and fruit quality.',
        'symptoms': [
            'Dark green leaves without spots or discoloration',
            'Uniform leaf size and shape',
            'Strong, upright stems supporting the plant structure',
            'Firm, glossy fruit with thick walls',
            'Consistent flowering and fruit set'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications based on soil tests',
                'Compost tea applications to boost soil biology',
                'Seaweed extract foliar sprays for micronutrients',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Calcium supplements to prevent blossom end rot',
                'Preventative fungicide program during high-risk periods',
                'Foliar micronutrient sprays as needed'
            ]
        },
        'prevention': [
            'Proper plant spacing (18-24 inches) for air circulation',
            'Mulching to retain moisture and suppress weeds',
            'Support structures for heavy-fruiting varieties',
            'Regular monitoring for early pest and disease detection',
            'Crop rotation with non-solanaceous crops'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Consistent soil moisture to prevent blossom end rot',
                'Drip irrigation to deliver water efficiently',
                'Morning watering to reduce evaporation',
                'Avoid overhead irrigation which can promote disease'
            ]
        },
        'environmental_factors': {
            'temperature': '70-85°F (21-29°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best fruit development'
        },
        'crop_impact': {
            'yield_potential': '5-10 pounds per plant in optimal conditions',
            'quality_characteristics': 'Firm, thick-walled fruit with vibrant color',
            'economic_value': 'Maximum marketability and shelf life'
        }
    },
    
    'Potato___Early_blight': {
        'name': 'Potato Early Blight',
        'scientific_name': 'Alternaria solani',
        'description': 'A fungal disease affecting potato plants that causes dark, target-like spots on leaves. Severe infections can lead to defoliation, reduced tuber size, and yield losses.',
        'symptoms': [
            'Dark brown to black target-like spots with concentric rings on leaves',
            'Yellowing of leaf tissue surrounding lesions',
            'Lesions that coalesce causing leaf death',
            'Lower leaves affected first with upward progression',
            'Shallow, dark lesions on tubers in severe cases'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively',
                'Sulfur applications at 7-10 day intervals',
                'Bacillus subtilis biological fungicide treatments',
                'Potassium bicarbonate sprays for light infections'
            ],
            'chemical': [
                'Chlorothalonil applications at 7-10 day intervals',
                'Azoxystrobin treatments for active infections',
                'Boscalid + pyraclostrobin combination products',
                'Mancozeb applications during high-risk periods'
            ]
        },
        'prevention': [
            'Plant certified disease-free seed potatoes',
            'Practice crop rotation with non-solanaceous crops for 2-3 years',
            'Maintain adequate plant spacing for air circulation',
            'Hill soil around plants to protect tubers',
            'Remove and destroy infected plant debris after harvest'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water deeply but infrequently to promote root development',
                'Avoid overhead irrigation which promotes fungal spread',
                'Water in the morning so foliage dries quickly'
            ]
        },
        'environmental_factors': {
            'temperature': '75-85°F (24-29°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Alternating wet and dry periods favor disease development'
        },
        'crop_impact': {
            'yield_reduction': '20-30% in severe cases',
            'quality_impact': 'Reduced tuber size and storage quality',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of foliage'
        }
    },
    
    'Potato___Late_blight': {
        'name': 'Potato Late Blight',
        'scientific_name': 'Phytophthora infestans',
        'description': 'A devastating oomycete disease affecting potato plants that causes water-soaked lesions on leaves and stems, and tuber rot. This is the disease responsible for the Irish Potato Famine and requires aggressive management.',
        'symptoms': [
            'Pale green to brown water-soaked lesions on leaves, often at leaf tips or margins',
            'White, fuzzy growth on the undersides of leaves in humid conditions',
            'Dark brown to purplish-black lesions on stems',
            'Rapid collapse of foliage in humid conditions',
            'Reddish-brown, granular rot in tubers that may appear externally as purple-brown areas'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively before disease onset',
                'More frequent applications (5-7 days) during high-risk periods',
                'Bacillus subtilis biological treatments as preventatives',
                'Immediate removal and destruction of infected plants'
            ],
            'chemical': [
                'Chlorothalonil applications preventatively',
                'Mefenoxam + chlorothalonil combinations for systemic protection',
                'Cymoxanil + famoxadone for curative and protective action',
                'Propamocarb + fluopicolide for systemic and translaminar activity'
            ]
        },
        'prevention': [
            'Plant certified disease-free seed potatoes',
            'Use resistant varieties when available',
            'Practice crop rotation with non-solanaceous crops for 3-4 years',
            'Monitor and destroy volunteer potatoes and solanaceous weeds',
            'Plant in well-drained soil with good air circulation'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Avoid overhead irrigation which promotes disease spread',
                'Water in the morning so foliage dries quickly',
                'Ensure good field drainage to prevent standing water'
            ]
        },
        'environmental_factors': {
            'temperature': '60-70°F (15-21°C) optimal for disease development',
            'humidity': 'High humidity (>90%) essential for infection',
            'conditions': 'Cool, wet weather with morning fog or dew promotes epidemics'
        },
        'crop_impact': {
            'yield_reduction': '40-100% in severe cases',
            'quality_impact': 'Tuber rot can continue in storage',
            'economic_threshold': 'Zero tolerance - preventative treatment required when conditions favor disease'
        }
    },
        'Potato___healthy': {
        'name': 'Healthy Potato',
        'scientific_name': 'Solanum tuberosum',
        'description': 'Healthy potato plants display dark green leaves, strong stems, and produce well-formed tubers. Proper maintenance ensures plant vigor, optimal yield, and high-quality tubers.',
        'symptoms': [
            'Dark green leaves without spots or discoloration',
            'Uniform leaf size and shape',
            'Strong, upright stems supporting the plant structure',
            'Consistent flowering and tuber development',
            'Well-formed, smooth-skinned tubers at harvest'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications based on soil tests',
                'Compost incorporation before planting',
                'Seaweed extract foliar sprays for micronutrients',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Split nitrogen applications for optimal utilization',
                'Preventative fungicide program during high-risk periods',
                'Micronutrient supplements as indicated by tissue tests'
            ]
        },
        'prevention': [
            'Plant certified disease-free seed potatoes',
            'Practice crop rotation with non-solanaceous crops',
            'Proper plant spacing (10-12 inches) for air circulation',
            'Hill soil around plants to protect developing tubers',
            'Regular monitoring for early pest and disease detection'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Consistent soil moisture during tuber formation',
                'Reduce irrigation as plants mature to allow proper skin set',
                'Drip irrigation to deliver water efficiently',
                'Avoid overhead irrigation which can promote disease'
            ]
        },
        'environmental_factors': {
            'temperature': '60-70°F (15-21°C) optimal for tuber development',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best growth'
        },
        'crop_impact': {
            'yield_potential': '300-500 cwt per acre in commercial production',
            'quality_characteristics': 'Uniform size, smooth skin, proper specific gravity',
            'economic_value': 'Maximum marketability and storage potential'
        }
    },
    
    'Raspberry___healthy': {
        'name': 'Healthy Raspberry',
        'scientific_name': 'Rubus idaeus',
        'description': 'Healthy raspberry plants display vibrant green leaves, strong canes, and produce plump, well-colored berries. Proper maintenance ensures plant vigor, productivity, and fruit quality.',
        'symptoms': [
            'Vibrant green leaves without spots or discoloration',
            'Strong, upright primocanes and floricanes',
            'Uniform leaf size and shape',
            'Plump, well-colored berries that separate easily from the receptacle',
            'Consistent annual growth of new canes'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications in spring',
                'Compost applications to improve soil health',
                'Seaweed extract foliar sprays for micronutrients',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Preventative fungicide program during wet periods',
                'Dormant oil sprays during winter',
                'Micronutrient supplements as indicated by tissue tests'
            ]
        },
        'prevention': [
            'Proper pruning to remove spent floricanes after fruiting',
            'Thinning of primocanes to 4-5 strong canes per foot of row',
            'Trellising to improve air circulation and light penetration',
            'Mulching to retain moisture and suppress weeds',
            'Regular monitoring for early pest and disease detection'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Drip irrigation to deliver water efficiently',
                'Consistent soil moisture during fruit development',
                'Avoid overhead irrigation which can promote disease',
                'Mulching to conserve soil moisture'
            ]
        },
        'environmental_factors': {
            'temperature': '60-75°F (15-24°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun to partial shade, with protection from harsh afternoon sun in hot climates'
        },
        'crop_impact': {
            'yield_potential': '5-10 pounds per plant for summer-bearing; 1-2 pounds for fall-bearing',
            'quality_characteristics': 'Firm, well-colored berries with good flavor balance',
            'economic_value': 'Maximum marketability and shelf life'
        }
    },
    
    'Soybean___healthy': {
        'name': 'Healthy Soybean',
        'scientific_name': 'Glycine max',
        'description': 'Healthy soybean plants display vibrant green leaves, strong stems, and produce well-filled pods. Proper maintenance ensures plant vigor, optimal yield, and high-quality beans.',
        'symptoms': [
            'Vibrant green trifoliate leaves without spots or discoloration',
            'Strong, upright stems with appropriate branching',
            'Uniform leaf size and shape',
            'Well-formed pods with 2-3 beans per pod',
            'Consistent nodulation on roots for nitrogen fixation'
        ],
        'treatment': {
            'organic': [
                'Proper inoculation with Bradyrhizobium japonicum at planting',
                'Balanced organic fertilizer applications based on soil tests',
                'Compost incorporation before planting',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Seed treatment with appropriate inoculant',
                'Balanced fertilizer applications based on soil tests',
                'Preventative fungicide program during high-risk periods',
                'Foliar micronutrient sprays as needed'
            ]
        },
        'prevention': [
            'Crop rotation with non-legume crops',
            'Proper plant spacing for air circulation',
            'Planting at optimal soil temperatures (55-60°F)',
            'Seed treatment with fungicides and inoculants',
            'Regular monitoring for early pest and disease detection'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Critical water needs during flowering and pod fill',
                'Drip or furrow irrigation to deliver water efficiently',
                'Avoid overhead irrigation which can promote disease',
                'Ensure good field drainage to prevent waterlogging'
            ]
        },
        'environmental_factors': {
            'temperature': '70-85°F (21-29°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best growth'
        },
        'crop_impact': {
            'yield_potential': '40-80 bushels per acre in optimal conditions',
            'quality_characteristics': 'Uniform seed size, appropriate protein and oil content',
            'economic_value': 'Maximum marketability and processing quality'
        }
    },
    
    'Squash___Powdery_mildew': {
        'name': 'Squash Powdery Mildew',
        'scientific_name': 'Podosphaera xanthii (formerly Sphaerotheca fuliginea) and Erysiphe cichoracearum',
        'description': 'A fungal disease affecting squash and other cucurbits that appears as a white powdery coating on leaves and stems. Severe infections can reduce photosynthesis, stunt growth, and decrease fruit quality and yield.',
        'symptoms': [
            'White powdery coating on leaves, stems, and sometimes fruit',
            'Initial symptoms appear as small white spots that expand to cover leaf surfaces',
            'Yellowing and premature death of affected leaves',
            'Stunted plant growth in severe cases',
            'Reduced fruit size and quality'
        ],
        'treatment': {
            'organic': [
                'Sulfur applications at 7-10 day intervals',
                'Potassium bicarbonate sprays for active infections',
                'Neem oil treatments every 7-14 days',
                'Milk spray (1:9 milk to water ratio) as a preventative'
            ],
            'chemical': [
                'Myclobutanil applications at first sign of disease',
                'Trifloxystrobin sprays at 7-14 day intervals',
                'Chlorothalonil treatments as a preventative',
                'Azoxystrobin applications for severe infections'
            ]
        },
        'prevention': [
            'Plant resistant varieties when available',
            'Proper plant spacing for air circulation',
            'Avoid overhead irrigation which increases humidity',
            'Remove and destroy infected plant parts',
            'Crop rotation with non-cucurbit crops'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water at the base of plants in the morning',
                'Ensure good soil drainage',
                'Avoid overwatering which can increase humidity around plants'
            ]
        },
        'environmental_factors': {
            'temperature': '68-80°F (20-27°C) optimal for disease development',
            'humidity': 'High humidity promotes infection, but free water inhibits spore germination',
            'conditions': 'Shade and poor air circulation increase disease severity'
        },
        'crop_impact': {
            'yield_reduction': '20-30% in severe cases',
            'quality_impact': 'Reduced fruit size and shortened harvest period',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of foliage'
        }
    },
        'Strawberry___Leaf_scorch': {
        'name': 'Strawberry Leaf Scorch',
        'scientific_name': 'Diplocarpon earlianum',
        'description': 'A fungal disease affecting strawberry plants that causes purple to red spots on leaves, which eventually develop necrotic centers. Severe infections can lead to leaf death, reduced plant vigor, and decreased fruit yield and quality.',
        'symptoms': [
            'Small, purple to red spots on upper leaf surfaces',
            'Spots that develop tan to gray centers as they enlarge',
            'Irregular, dark purple borders around lesions',
            'Leaves that appear scorched as lesions coalesce',
            'Premature leaf death in severe cases'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively',
                'Sulfur applications at 7-10 day intervals',
                'Potassium bicarbonate sprays for light infections',
                'Bacillus subtilis biological fungicide treatments'
            ],
            'chemical': [
                'Captan applications at 7-14 day intervals',
                'Myclobutanil treatments during active growth',
                'Azoxystrobin sprays for established infections',
                'Thiophanate-methyl applications during the growing season'
            ]
        },
        'prevention': [
            'Plant resistant varieties when available',
            'Use disease-free transplants',
            'Practice crop rotation with non-strawberry crops for 2-3 years',
            'Remove and destroy infected leaves during the growing season',
            'Renovate beds immediately after harvest'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water at the base of plants in the morning',
                'Ensure good soil drainage',
                'Avoid overhead irrigation which promotes fungal spread'
            ]
        },
        'environmental_factors': {
            'temperature': '68-86°F (20-30°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Prolonged leaf wetness (6+ hours) significantly increases infection risk'
        },
        'crop_impact': {
            'yield_reduction': '20-30% in severe cases',
            'quality_impact': 'Reduced fruit size and sugar content',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of foliage'
        }
    },
    
    'Strawberry___healthy': {
        'name': 'Healthy Strawberry',
        'scientific_name': 'Fragaria × ananassa',
        'description': 'Healthy strawberry plants display vibrant green leaves, strong crowns, and produce firm, well-colored berries. Proper maintenance ensures plant vigor, productivity, and fruit quality.',
        'symptoms': [
            'Vibrant green trifoliate leaves without spots or discoloration',
            'Strong, compact crown with multiple crowns developing in mature plants',
            'Uniform leaf size and shape',
            'Firm, well-colored berries with seeds (achenes) evenly distributed',
            'Vigorous runners in appropriate quantities for the variety'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications based on soil tests',
                'Compost applications to improve soil health',
                'Seaweed extract foliar sprays for micronutrients',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Preventative fungicide program during wet periods',
                'Calcium supplements to improve fruit firmness',
                'Micronutrient supplements as indicated by tissue tests'
            ]
        },
        'prevention': [
            'Plant certified disease-free transplants',
            'Proper plant spacing (12-18 inches) for air circulation',
            'Mulching with straw or plastic to keep fruit clean and reduce disease',
            'Renovation of beds after harvest',
            'Regular monitoring for early pest and disease detection'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Drip irrigation to deliver water efficiently',
                'Consistent soil moisture during fruit development',
                'Avoid overhead irrigation which can promote disease',
                'Mulching to conserve soil moisture'
            ]
        },
        'environmental_factors': {
            'temperature': '60-75°F (15-24°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best fruit development'
        },
        'crop_impact': {
            'yield_potential': '1-2 pounds per plant in first fruiting year',
            'quality_characteristics': 'Firm, sweet berries with good size and color',
            'economic_value': 'Maximum marketability and shelf life'
        }
    },
    
    'Tomato___Bacterial_spot': {
        'name': 'Tomato Bacterial Spot',
        'scientific_name': 'Xanthomonas campestris pv. vesicatoria',
        'description': 'A bacterial disease affecting tomato plants that causes spots on leaves, stems, and fruit. Severe infections can lead to defoliation, sunscald of fruit, and significant yield losses.',
        'symptoms': [
            'Small, water-soaked spots on leaves that become brown with yellow halos',
            'Spots with a greasy appearance on fruit that become scabby',
            'Defoliation in severe cases exposing fruit to sunscald',
            'Dark, elongated lesions on stems and petioles',
            'Spots that may tear or fall out, giving leaves a tattered appearance'
        ],
        'treatment': {
            'organic': [
                'Copper-based bactericides applied preventatively',
                'Bacillus subtilis biological treatments',
                'Hydrogen dioxide products for surface sterilization',
                'Crop rotation with non-solanaceous crops for 2-3 years'
            ],
            'chemical': [
                'Copper-based bactericides applied at 7-10 day intervals',
                'Copper-mancozeb combinations for better efficacy',
                'Acibenzolar-S-methyl (Actigard) as a plant activator',
                'Streptomycin applications in seedling production (where legal)'
            ]
        },
        'prevention': [
            'Plant certified disease-free seeds and transplants',
            'Use resistant varieties when available',
            'Avoid overhead irrigation which promotes bacterial spread',
            'Practice crop rotation with non-solanaceous crops',
            'Remove and destroy infected plant debris after harvest'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water at the base of plants in the morning',
                'Ensure good soil drainage',
                'Avoid working with plants when foliage is wet'
            ]
        },
        'environmental_factors': {
            'temperature': '75-86°F (24-30°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Rainfall, overhead irrigation, and wind-driven rain increase spread'
        },
        'crop_impact': {
            'yield_reduction': '10-50% in severe cases',
            'quality_impact': 'Reduced marketability due to fruit blemishes',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of plants'
        }
    },
    
    'Tomato___Early_blight': {
        'name': 'Tomato Early Blight',
        'scientific_name': 'Alternaria solani',
        'description': 'A fungal disease affecting tomato plants that causes dark, target-like spots on leaves. Severe infections can lead to defoliation, reduced fruit size, and yield losses.',
        'symptoms': [
            'Dark brown to black target-like spots with concentric rings on leaves',
            'Yellowing of leaf tissue surrounding lesions',
            'Lesions that coalesce causing leaf death',
            'Lower leaves affected first with upward progression',
            'Dark, sunken lesions with concentric rings on fruit, usually at the stem end'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively',
                'Sulfur applications at 7-10 day intervals',
                'Bacillus subtilis biological fungicide treatments',
                'Potassium bicarbonate sprays for light infections'
            ],
            'chemical': [
                'Chlorothalonil applications at 7-10 day intervals',
                'Azoxystrobin treatments for active infections',
                'Boscalid + pyraclostrobin combination products',
                'Mancozeb applications during high-risk periods'
            ]
        },
        'prevention': [
            'Plant resistant varieties when available',
            'Practice crop rotation with non-solanaceous crops for 2-3 years',
            'Maintain adequate plant spacing for air circulation',
            'Stake or cage plants to keep foliage off the ground',
            'Remove and destroy infected plant debris after harvest'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water deeply but infrequently to promote root development',
                'Avoid overhead irrigation which promotes fungal spread',
                'Water in the morning so foliage dries quickly'
            ]
        },
        'environmental_factors': {
            'temperature': '75-85°F (24-29°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Alternating wet and dry periods favor disease development'
        },
        'crop_impact': {
            'yield_reduction': '20-30% in severe cases',
            'quality_impact': 'Reduced fruit size and marketability',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of foliage'
        }
    },
        'Tomato___Late_blight': {
        'name': 'Tomato Late Blight',
        'scientific_name': 'Phytophthora infestans',
        'description': 'A devastating oomycete disease affecting tomato plants that causes water-soaked lesions on leaves and stems, and fruit rot. This is the same pathogen that caused the Irish Potato Famine and requires aggressive management.',
        'symptoms': [
            'Pale green to brown water-soaked lesions on leaves, often at leaf tips or margins',
            'White, fuzzy growth on the undersides of leaves in humid conditions',
            'Dark brown to black lesions on stems',
            'Rapid collapse of foliage in humid conditions',
            'Firm, brown, greasy-looking lesions on fruit that may cover large areas'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively before disease onset',
                'More frequent applications (5-7 days) during high-risk periods',
                'Bacillus subtilis biological treatments as preventatives',
                'Immediate removal and destruction of infected plants'
            ],
            'chemical': [
                'Chlorothalonil applications preventatively',
                'Mefenoxam + chlorothalonil combinations for systemic protection',
                'Cymoxanil + famoxadone for curative and protective action',
                'Propamocarb + fluopicolide for systemic and translaminar activity'
            ]
        },
        'prevention': [
            'Plant resistant varieties when available',
            'Practice crop rotation with non-solanaceous crops for 3-4 years',
            'Maintain adequate plant spacing for air circulation',
            'Stake or cage plants to keep foliage off the ground',
            'Monitor and destroy volunteer tomatoes and solanaceous weeds'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Avoid overhead irrigation which promotes disease spread',
                'Water in the morning so foliage dries quickly',
                'Ensure good field drainage to prevent standing water'
            ]
        },
        'environmental_factors': {
            'temperature': '60-70°F (15-21°C) optimal for disease development',
            'humidity': 'High humidity (>90%) essential for infection',
            'conditions': 'Cool, wet weather with morning fog or dew promotes epidemics'
        },
        'crop_impact': {
            'yield_reduction': '40-100% in severe cases',
            'quality_impact': 'Fruit rot renders produce unmarketable',
            'economic_threshold': 'Zero tolerance - preventative treatment required when conditions favor disease'
        }
    },
    
    'Tomato___Leaf_Mold': {
        'name': 'Tomato Leaf Mold',
        'scientific_name': 'Passalora fulva (formerly Fulvia fulva or Cladosporium fulvum)',
        'description': 'A fungal disease primarily affecting greenhouse tomatoes that causes yellow spots on the upper leaf surface and olive-green to grayish-brown mold on the lower surface. Severe infections can lead to defoliation and reduced yield.',
        'symptoms': [
            'Pale green or yellow spots on upper leaf surfaces',
            'Olive-green to grayish-brown velvety mold on lower leaf surfaces',
            'Leaves that curl, wither, and drop prematurely',
            'Occasionally affects stems, blossoms, and fruit',
            'Upward progression of symptoms from lower to upper leaves'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively',
                'Sulfur applications at 7-10 day intervals',
                'Potassium bicarbonate sprays for light infections',
                'Bacillus subtilis biological fungicide treatments'
            ],
            'chemical': [
                'Chlorothalonil applications at 7-14 day intervals',
                'Mancozeb treatments during early disease development',
                'Difenoconazole + cyprodinil for established infections',
                'Azoxystrobin applications for protective and curative action'
            ]
        },
        'prevention': [
            'Plant resistant varieties when available',
            'Maintain adequate plant spacing for air circulation',
            'Improve greenhouse ventilation to reduce humidity',
            'Remove and destroy infected leaves during the growing season',
            'Practice crop rotation in greenhouse settings'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water at the base of plants in the morning',
                'Avoid overhead irrigation which promotes fungal spread',
                'Reduce humidity in greenhouse environments with proper ventilation'
            ]
        },
        'environmental_factors': {
            'temperature': '70-75°F (21-24°C) optimal for disease development',
            'humidity': 'High humidity (>85%) essential for infection',
            'conditions': 'Common in greenhouse environments with poor air circulation'
        },
        'crop_impact': {
            'yield_reduction': '10-30% in severe cases',
            'quality_impact': 'Reduced fruit size due to decreased photosynthesis',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of foliage'
        }
    },
    
    'Tomato___Septoria_leaf_spot': {
        'name': 'Septoria Leaf Spot',
        'scientific_name': 'Septoria lycopersici',
        'description': 'A fungal disease affecting tomato plants that causes small, circular spots with dark borders and light centers on leaves. Severe infections can lead to significant defoliation and reduced yield.',
        'symptoms': [
            'Small, circular spots (1/16 to 1/8 inch) with dark borders and light gray or tan centers',
            'Tiny black fruiting bodies (pycnidia) visible in the center of mature spots',
            'Lower leaves affected first with upward progression',
            'Yellowing and dropping of severely infected leaves',
            'Rarely affects stems, blossoms, or fruit'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively',
                'Sulfur applications at 7-10 day intervals',
                'Potassium bicarbonate sprays for light infections',
                'Bacillus subtilis biological fungicide treatments'
            ],
            'chemical': [
                'Chlorothalonil applications at 7-10 day intervals',
                'Mancozeb treatments during early disease development',
                'Azoxystrobin sprays for established infections',
                'Propiconazole applications during the growing season'
            ]
        },
        'prevention': [
            'Practice crop rotation with non-solanaceous crops for 2-3 years',
            'Remove and destroy infected plant debris after harvest',
            'Maintain adequate plant spacing for air circulation',
            'Stake or cage plants to keep foliage off the ground',
            'Mulch around plants to prevent soil splash onto leaves'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water at the base of plants in the morning',
                'Avoid overhead irrigation which promotes fungal spread',
                'Mulch to reduce soil splash during rainfall'
            ]
        },
        'environmental_factors': {
            'temperature': '68-77°F (20-25°C) optimal for disease development',
            'humidity': 'High humidity promotes infection',
            'conditions': 'Rainfall, overhead irrigation, and leaf wetness for 48+ hours increases infection risk'
        },
        'crop_impact': {
            'yield_reduction': '15-30% in severe cases',
            'quality_impact': 'Reduced fruit size due to decreased photosynthesis',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of foliage'
        }
    },
    
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'name': 'Two-spotted Spider Mite',
        'scientific_name': 'Tetranychus urticae',
        'description': 'A common pest affecting tomato plants that causes stippling and yellowing of leaves. Severe infestations can lead to leaf drop, reduced photosynthesis, and decreased yield. Spider mites are not a disease but an arthropod pest.',
        'symptoms': [
            'Fine stippling or speckling on upper leaf surfaces',
            'Yellowing or bronzing of leaves as feeding progresses',
            'Fine webbing on the undersides of leaves and between plant parts in heavy infestations',
            'Tiny moving dots (mites) visible with a hand lens, particularly on leaf undersides',
            'Premature leaf drop in severe cases'
        ],
        'treatment': {
            'organic': [
                'Insecticidal soap applications targeting leaf undersides',
                'Neem oil treatments at 7-10 day intervals',
                'Predatory mites (Phytoseiulus persimilis) releases',
                'Sulfur applications (note: may harm beneficial mites)'
            ],
            'chemical': [
                'Abamectin applications targeting leaf undersides',
                'Bifenazate treatments for quick knockdown',
                'Spiromesifen for all life stages control',
                'Fenpyroximate applications with thorough coverage'
            ]
        },
        'prevention': [
            'Regular monitoring with a hand lens, especially during hot, dry conditions',
            'Maintain adequate plant moisture to reduce plant stress',
            'Avoid excessive nitrogen fertilization which promotes mite reproduction',
            'Introduce and conserve natural predators',
            'Avoid broad-spectrum insecticides which kill beneficial predators'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Avoid water stress which increases plant susceptibility',
                'Occasional overhead irrigation or spraying with water can reduce populations',
                'Maintain consistent soil moisture',
                'Increase humidity in greenhouse environments to discourage mites'
            ]
        },
        'environmental_factors': {
            'temperature': '80-90°F (27-32°C) optimal for mite reproduction',
            'humidity': 'Low humidity (below 50%) favors mite population growth',
            'conditions': 'Hot, dry conditions lead to rapid population increases'
        },
        'crop_impact': {
            'yield_reduction': '10-40% in severe cases',
            'quality_impact': 'Reduced fruit size and sugar content',
            'economic_threshold': 'Treatment warranted when 25-30% of leaves show damage or 10-15 mites per leaflet'
        }
    },
    
    'Tomato___Target_Spot': {
        'name': 'Tomato Target Spot',
        'scientific_name': 'Corynespora cassiicola',
        'description': 'A fungal disease affecting tomato plants that causes circular lesions with concentric rings resembling targets on leaves, stems, and fruit. Severe infections can lead to defoliation and significant yield losses.',
        'symptoms': [
            'Brown to dark brown circular lesions with concentric rings on leaves',
            'Lesions that may have yellow halos and eventually dry out and tear',
            'Similar target-like lesions on stems and petioles',
            'Sunken, dark brown, leathery lesions on fruit',
            'Premature defoliation in severe cases'
        ],
        'treatment': {
            'organic': [
                'Copper-based fungicides applied preventatively',
                'Sulfur applications at 7-10 day intervals',
                'Bacillus subtilis biological fungicide treatments',
                'Potassium bicarbonate sprays for light infections'
            ],
            'chemical': [
                'Chlorothalonil applications at 7-10 day intervals',
                'Azoxystrobin treatments for established infections',
                'Fluxapyroxad + pyraclostrobin combination products',
                'Difenoconazole + cyprodinil for protective and curative action'
            ]
        },
        'prevention': [
            'Practice crop rotation with non-solanaceous crops for 2-3 years',
            'Maintain adequate plant spacing for air circulation',
            'Stake or cage plants to keep foliage off the ground',
            'Remove and destroy infected plant debris after harvest',
            'Avoid overhead irrigation which promotes fungal spread'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Use drip irrigation to keep foliage dry',
                'Water at the base of plants in the morning',
                'Ensure good soil drainage',
                'Mulch to reduce soil splash during rainfall'
            ]
        },
        'environmental_factors': {
            'temperature': '70-80°F (21-27°C) optimal for disease development',
            'humidity': 'High humidity (>80%) promotes infection',
            'conditions': 'Prolonged leaf wetness (16+ hours) significantly increases infection risk'
        },
        'crop_impact': {
            'yield_reduction': '20-40% in severe cases',
            'quality_impact': 'Reduced marketability due to fruit lesions',
            'economic_threshold': 'Treatment warranted when disease appears on 5% of foliage'
        }
    },
    
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus',
        'scientific_name': 'Tomato yellow leaf curl virus (TYLCV)',
        'description': 'A devastating viral disease transmitted by whiteflies that causes stunting, leaf curling, and significant yield losses. Once infected, plants cannot be cured, making prevention critical.',
        'symptoms': [
            'Upward curling and cupping of leaves',
            'Leaves that become small and yellow, particularly at plant tops',
            'Severe stunting of plants with a bushy appearance',
            'Flowers that may drop before fruit set',
            'Significantly reduced fruit production'
        ],
        'treatment': {
            'organic': [
                'No treatments cure infected plants',
                'Insecticidal soaps for whitefly control',
                'Neem oil applications for whitefly suppression',
                'Yellow sticky traps to monitor and reduce whitefly populations'
            ],
            'chemical': [
                'No treatments cure infected plants',
                'Systemic insecticides like imidacloprid for whitefly control',
                'Pyrethroids for adult whitefly management',
                'Insect growth regulators for whitefly nymph control'
            ]
        },
        'prevention': [
            'Plant resistant varieties (look for TYLCV resistance)',
            'Use reflective mulches to repel whiteflies',
            'Install fine mesh screens in greenhouse production',
            'Remove and destroy infected plants immediately',
            'Maintain a whitefly-free period between crop cycles'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Maintain consistent soil moisture to reduce plant stress',
                'Avoid water stress which can exacerbate symptoms',
                'Use drip irrigation for efficient water delivery',
                'Ensure good drainage to promote overall plant health'
            ]
        },
        'environmental_factors': {
            'temperature': 'Virus replication and symptom expression increase at 80-90°F (27-32°C)',
            'humidity': 'Whitefly populations higher in moderate to high humidity',
            'conditions': 'Year-round pressure in tropical/subtropical climates'
        },
        'crop_impact': {
            'yield_reduction': '50-100% in severe cases',
            'quality_impact': 'Few or no marketable fruit produced',
            'economic_threshold': 'Zero tolerance - prevention is critical as there is no cure'
        }
    },
        'Tomato___Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus',
        'scientific_name': 'Tomato mosaic virus (ToMV) and Tobacco mosaic virus (TMV)',
        'description': 'A viral disease affecting tomato plants that causes mottling and distortion of leaves, stunted growth, and fruit discoloration. The virus is highly stable and can persist in soil and plant debris for years.',
        'symptoms': [
            'Light and dark green mottling or mosaic pattern on leaves',
            'Leaves that may be distorted, curled, or reduced in size',
            'Stunted plant growth and reduced vigor',
            'Yellow blotches, rings, or necrotic spots on fruit',
            'Internal fruit browning or necrosis'
        ],
        'treatment': {
            'organic': [
                'No treatments cure infected plants',
                'Remove and destroy infected plants immediately',
                'Milk spray (20% milk solution) on hands and tools for disinfection',
                'Maintain optimal growing conditions to reduce stress'
            ],
            'chemical': [
                'No chemical treatments cure infected plants',
                'Disinfect tools with 10% bleach solution or 70% alcohol',
                'Use commercial disinfectants for greenhouse surfaces',
                'Trisodium phosphate (TSP) for seed treatment'
            ]
        },
        'prevention': [
            'Plant certified virus-free seeds and transplants',
            'Use resistant varieties (look for Tm-2 or Tm-22 resistance genes)',
            'Practice strict sanitation with tools and hands when handling plants',
            'Control perennial weeds that may harbor the virus',
            'Avoid tobacco products when working with plants (TMV can be carried on tobacco)'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Maintain consistent soil moisture to reduce plant stress',
                'Avoid water stress which can exacerbate symptoms',
                'Use drip irrigation to minimize plant handling',
                'Ensure good drainage to promote overall plant health'
            ]
        },
        'environmental_factors': {
            'temperature': 'Symptoms most severe at 80-85°F (27-29°C)',
            'humidity': 'Not directly affected by humidity',
            'conditions': 'High light intensity can increase symptom severity'
        },
        'crop_impact': {
            'yield_reduction': '20-70% depending on infection timing and severity',
            'quality_impact': 'Reduced marketability due to fruit symptoms',
            'economic_threshold': 'Zero tolerance - prevention is critical as there is no cure'
        }
    },
    
    'Tomato___healthy': {
        'name': 'Healthy Tomato',
        'scientific_name': 'Solanum lycopersicum',
        'description': 'Healthy tomato plants display vibrant green leaves, strong stems, and produce firm, well-colored fruit. Proper maintenance ensures plant vigor, productivity, and fruit quality.',
        'symptoms': [
            'Vibrant green leaves without spots or discoloration',
            'Strong, upright stems with appropriate branching',
            'Uniform leaf size and shape',
            'Consistent flowering and fruit set',
            'Firm, well-developed fruit with appropriate color for variety'
        ],
        'treatment': {
            'organic': [
                'Balanced organic fertilizer applications based on soil tests',
                'Compost tea applications to boost soil biology',
                'Seaweed extract foliar sprays for micronutrients',
                'Beneficial insect releases for pest management'
            ],
            'chemical': [
                'Balanced NPK fertilizer applications based on soil tests',
                'Calcium supplements to prevent blossom end rot',
                'Preventative fungicide program during high-risk periods',
                'Micronutrient supplements as indicated by tissue tests'
            ]
        },
        'prevention': [
            'Proper plant spacing (18-24 inches) for air circulation',
            'Staking, caging, or trellising to keep plants off ground',
            'Mulching to retain moisture and suppress weeds',
            'Pruning to improve air circulation and light penetration',
            'Regular monitoring for early pest and disease detection'
        ],
        'water_management': {
            'requirements': 'Moderate - 1-2 inches per week during growing season',
            'recommendations': [
                'Consistent soil moisture to prevent blossom end rot and cracking',
                'Drip irrigation to deliver water efficiently',
                'Morning watering to reduce evaporation and disease risk',
                'Avoid overhead irrigation which can promote disease'
            ]
        },
        'environmental_factors': {
            'temperature': '65-85°F (18-29°C) optimal for growth',
            'humidity': 'Moderate humidity preferred',
            'conditions': 'Full sun (6-8 hours daily) for best fruit development'
        },
        'crop_impact': {
            'yield_potential': '10-30 pounds per plant depending on variety and growing conditions',
            'quality_characteristics': 'Firm fruit with good flavor, color, and shelf life',
            'economic_value': 'Maximum marketability and consumer appeal'
        }
    }
}

# Custom JSON encoder to handle undefined values
class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif str(obj).lower() == 'undefined' or obj is None:
                return None
            return super().default(obj)
        except:
            return None

# Helper functions
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_disease_info(disease_class):
    """Return information about the disease from the database"""
    # Return disease info if available, otherwise return generic info
    return disease_database.get(disease_class, {
        'name': disease_class.replace('___', ' - ').replace('_', ' '),
        'description': 'Detailed information not available for this condition.',
        'treatment': 'Consult with a local agricultural expert for treatment options.',
        'prevention': 'Follow general crop management best practices.'
    })

def visualize_prediction(img_path, output_path, result):
    """Create a visualization of the prediction on the image."""
    try:
        # Load the original image
        img = cv2.imread(img_path)
        if img is None:
            # Try with PIL if OpenCV fails
            pil_img = Image.open(img_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Resize for display if needed
        display_img = cv2.resize(img, (640, 480)) if img.shape[0] > 480 or img.shape[1] > 640 else img.copy()
        
        # Format the label
        class_name = str(result['class']).replace('___', ' - ').replace('_', ' ')
        confidence = float(result['confidence']) * 100
        label = f"{class_name} ({confidence:.1f}%)"
        
        # Add a semi-transparent overlay at the bottom
        overlay = display_img.copy()
        overlay_height = 60
        h, w = display_img.shape[:2]
        cv2.rectangle(overlay, (0, h-overlay_height), (w, h), (0, 0, 0), -1)
        
        # Correct way to use addWeighted
        alpha = 0.7
        beta = 0.3
        cv2.addWeighted(overlay, alpha, display_img, beta, 0, display_img)
        
        # Add text with prediction
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_color = (255, 255, 255)  # White text
        
        # Position text in the overlay
        text_position = (10, h - 25)
        cv2.putText(display_img, label, text_position, font, font_scale, text_color, font_thickness)
        
        # Save the visualization
        cv2.imwrite(output_path, display_img)
        return output_path
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        # Create a simple fallback visualization or just copy the original
        try:
            import shutil
            shutil.copy(img_path, output_path)
        except:
            # If even copying fails, create a blank image
            blank_img = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank_img, "Visualization failed", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(output_path, blank_img)
        return output_path
def ensure_serializable(obj):
    """Ensure all values in a dictionary are JSON serializable."""
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return ensure_serializable(obj.tolist())
    elif obj is None:
        return None
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Convert any other types to string to ensure they're serializable
        return str(obj)
    
def clean_old_uploads(max_age_days=7):
    """Clean up old uploaded files to save disk space"""
    try:
        current_time = time.time()
        count = 0
        for filename in os.listdir(UPLOAD_FOLDER):
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file_age_days = (current_time - os.path.getmtime(filepath)) / (60*60*24)
            if file_age_days > max_age_days:
                os.remove(filepath)
                count += 1
        if count > 0:
            logger.info(f"Cleaned up {count} old files from uploads folder")
    except Exception as e:
        logger.error(f"Error cleaning old uploads: {str(e)}")

def debug_json(obj, path=""):
    """Debug function to identify non-serializable values in an object"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            current_path = f"{path}.{k}" if path else k
            debug_json(v, current_path)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            debug_json(item, f"{path}[{i}]")
    else:
        try:
            json.dumps(obj)
        except TypeError:
            logger.error(f"Non-serializable value at {path}: {type(obj)} - {obj}")
            print(f"⚠️ Non-serializable value at {path}: {type(obj)} - {obj}")

# Initialize the crop disease detector
try:
    from crop_detection import CropDiseaseDetector
    
    # Specify paths explicitly
    model_path = os.path.join('models', 'plant_disease_model_best.keras')
    class_indices_path = os.path.join('models', 'class_indices.json')
    
    # Check if files exist before attempting to load
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        print(f"⚠️ Model file not found: {model_path}")
        detector = None
    elif not os.path.exists(class_indices_path):
        logger.error(f"Class indices file not found: {class_indices_path}")
        print(f"⚠️ Class indices file not found: {class_indices_path}")
        detector = None
    else:
        # Initialize with explicit paths
        detector = CropDiseaseDetector(model_path, class_indices_path)
        logger.info("Successfully loaded disease detector model")
        print("✅ Loaded disease detector model")
except Exception as e:
    error_msg = f"Could not load disease detector model: {str(e)}"
    logger.error(error_msg)
    print(f"⚠️ {error_msg}")
    detector = None

# Routes
@app.route('/')
def index():
    """Render the home page"""
    # Clean old uploads periodically
    if not session.get('cleaned_uploads'):
        clean_old_uploads()
        session['cleaned_uploads'] = True
    
    return render_template('index.html', app_name="YOUR CROP MATTERS")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    # Security check to prevent directory traversal
    if '..' in filename or filename.startswith('/'):
        return "Invalid filename", 400
        
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Cache-Control'] = 'public, max-age=31536000'  # Cache for a year
    return response

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and make disease prediction"""
    # Check if model is available
    if detector is None:
        flash('Sorry, the disease detection model is currently unavailable. Please try again later.')
        
        # Handle AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return json.dumps({
                'success': False, 
                'error': 'SORRY, TRY AGAIN'
            }, cls=SafeJSONEncoder), 200, {'Content-Type': 'application/json'}
        
        return redirect(url_for('index'))
    
    # Check if a file was included in the request
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    
    file = request.files['file']
    
    # Check if a file was selected
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    # Process the file if it's valid
    if file and allowed_file(file.filename):
        # Create a unique filename to avoid collisions
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the uploaded file
        file.save(filepath)
        logger.info(f"Saved uploaded file: {filename}")
        
        # Make prediction
        try:
            # Get model prediction
            start_time = time.time()
            raw_result = detector.predict(filepath)
            
            # Convert result to a safe format for JSON serialization
            safe_result = {}
            
            # Handle class
            if 'class' in raw_result:
                safe_result['class'] = str(raw_result['class']) if raw_result['class'] is not None else "Unknown"
            else:
                safe_result['class'] = "Unknown"
                
            # Handle confidence
            if 'confidence' in raw_result:
                try:
                    safe_result['confidence'] = float(raw_result['confidence'])
                except (TypeError, ValueError):
                    safe_result['confidence'] = 0.0
            else:
                safe_result['confidence'] = 0.0
                
            # Handle all_probabilities
            safe_result['all_probabilities'] = {}
            if 'all_probabilities' in raw_result and isinstance(raw_result['all_probabilities'], dict):
                for k, v in raw_result['all_probabilities'].items():
                    try:
                        if v is not None and str(v).lower() != 'undefined':
                            safe_result['all_probabilities'][str(k)] = float(v)
                        else:
                            safe_result['all_probabilities'][str(k)] = 0.0
                    except (TypeError, ValueError):
                        safe_result['all_probabilities'][str(k)] = 0.0
            
            # Use the safe result for the rest of the function
            result = safe_result
            
            prediction_time = time.time() - start_time
            
            # Log the prediction
            logger.info(f"Prediction for {filename}: {result['class']} with confidence {result['confidence']:.4f} in {prediction_time:.2f}s")
            
            # Create visualization with prediction overlay
            vis_filename = f"vis_{filename}"
            vis_filepath = os.path.join(app.config['UPLOAD_FOLDER'], vis_filename)
            
            # Use the visualization function
            visualize_prediction(filepath, vis_filepath, result)
            
            # Get disease information from database
            disease_info = get_disease_info(result['class'])
            
            # Handle AJAX requests
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                response_data = {
                    'success': True,
                    'result': result,
                    'image_file': filename,
                    'vis_image': vis_filename,
                    'disease_info': disease_info,
                    'processing_time': f"{prediction_time:.2f}"
                }
                
                return json.dumps(response_data, cls=SafeJSONEncoder), 200, {'Content-Type': 'application/json'}
            
            # Render results page for normal requests
            return render_template('result.html', 
                                  app_name="YOUR CROP MATTERS",
                                  result=result,
                                  image_file=filename,
                                  vis_image=vis_filename,
                                  disease_info=disease_info,
                                  processing_time=f"{prediction_time:.2f}")
        except Exception as e:
            # Handle errors during prediction
            error_msg = 'SORRY, TRY AGAIN'
            logger.error(f"Error processing {filename}: {str(e)}")
            flash(error_msg)
            
            # Create a visualization with error message
            vis_filename = f"vis_{filename}"
            vis_filepath = os.path.join(app.config['UPLOAD_FOLDER'], vis_filename)
            
            # Create a simple error visualization
            try:
                img = cv2.imread(filepath)
                if img is None:
                    pil_img = Image.open(filepath)
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                
                # Resize for display if needed
                display_img = cv2.resize(img, (640, 480)) if img.shape[0] > 480 or img.shape[1] > 640 else img.copy()
                
                # Add a semi-transparent overlay
                overlay = display_img.copy()
                h, w = display_img.shape[:2]
                cv2.rectangle(overlay, (0, h-60), (w, h), (0, 0, 0), -1)
                
                # Blend the overlay
                alpha = 0.7
                beta = 0.3
                cv2.addWeighted(overlay, alpha, display_img, beta, 0, display_img)
                
                # Add error text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display_img, "SORRY, TRY AGAIN", (10, h-25), font, 0.7, (255, 255, 255), 2)
                
                cv2.imwrite(vis_filepath, display_img)
            except Exception as viz_error:
                logger.error(f"Error creating error visualization: {str(viz_error)}")
                # If visualization fails, create a blank image with error message
                blank_img = np.zeros((480, 640, 3), np.uint8)
                cv2.putText(blank_img, "SORRY, TRY AGAIN", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(vis_filepath, blank_img)
            
            # Handle AJAX requests
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return json.dumps({
                    'success': False, 
                    'error': error_msg,
                    'image_file': filename,
                    'vis_image': vis_filename
                }, cls=SafeJSONEncoder), 200, {'Content-Type': 'application/json'}
            
            # For regular requests, show the error on the result page
            return render_template('result.html',
                                  app_name="YOUR CROP MATTERS",
                                  error=error_msg,
                                  image_file=filename,
                                  vis_image=vis_filename)
    
    # Handle invalid file type
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
    
    # Handle AJAX requests
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return json.dumps({'success': False, 'error': 'Invalid file type'}, cls=SafeJSONEncoder), 200, {'Content-Type': 'application/json'}
    
    return redirect(url_for('index'))

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html', app_name="YOUR CROP MATTERS")

@app.route('/contact')
def contact():
    """Render the contact page"""
    return render_template('contact.html', app_name="YOUR CROP MATTERS")

@app.route('/api/health')
def health_check():
    """API endpoint for health checks"""
    return jsonify({
        'status': 'healthy' if detector is not None else 'degraded',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': detector is not None
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.path}")
    return render_template('404.html', app_name="YOUR CROP MATTERS"), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    logger.error(f"500 error: {str(e)}")
    return render_template('500.html', app_name="YOUR CROP MATTERS"), 500

# Run the application
if __name__ == '__main__':
    print("🌱 Starting YOUR CROP MATTERS application")
    print(f"📁 Upload directory: {os.path.abspath(UPLOAD_FOLDER)}")
    logger.info("Application started")
    
    # Start the Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)