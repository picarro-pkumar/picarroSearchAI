#!/usr/bin/env python3
"""
Demo Data Loader for Picarro SearchAI

This script loads permanent demo documents about Picarro analyzers into the knowledge base.
It can be run multiple times safely and will check for existing documents before adding new ones.
"""

import logging
import hashlib
from typing import Dict, Any, List
from doc_processor import DocumentProcessor
from ai_responder import AIResponder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_document_id(content: str, metadata: Dict[str, Any]) -> str:
    """
    Generate a unique document ID based on content and metadata.
    
    Args:
        content: Document content
        metadata: Document metadata
        
    Returns:
        Unique document ID
    """
    # Create a hash from content and key metadata
    hash_input = f"{content}:{metadata.get('product', '')}:{metadata.get('category', '')}"
    return hashlib.md5(hash_input.encode()).hexdigest()


def get_demo_documents() -> List[Dict[str, Any]]:
    """
    Get the list of demo documents to load.
    
    Returns:
        List of document dictionaries with content and metadata
    """
    return [
        {
            "id": "g2301_specs",
            "content": """
            Picarro's G2301 gas analyzer is a high-precision instrument designed for measuring carbon dioxide (CO2), methane (CH4), and water vapor (H2O) concentrations simultaneously. 
            
            Key Specifications:
            - Detection limits: CO2: 0.1 ppm, CH4: 0.5 ppb, H2O: 0.01%
            - Precision: CO2: 0.02 ppm, CH4: 0.1 ppb
            - Measurement rate: 1 Hz
            - Warm-up time: < 30 minutes
            - Operating temperature: 5-40°C
            - Power consumption: 150W typical
            
            The G2301 uses cavity ring-down spectroscopy (CRDS) technology to achieve parts-per-billion (ppb) detection limits. 
            It features automatic temperature and pressure compensation, real-time data logging, and can operate continuously for extended periods.
            
            Applications include atmospheric research, greenhouse gas monitoring, industrial emissions tracking, and environmental compliance monitoring.
            """,
            "metadata": {
                "product": "G2301",
                "category": "gas_analyzer",
                "technology": "CRDS",
                "applications": "atmospheric_research,greenhouse_gas_monitoring,industrial_emissions",
                "measurements": "CO2,CH4,H2O",
                "detection_limits": "ppb_level",
                "type": "specifications"
            }
        },
        {
            "id": "g2401_specs",
            "content": """
            The Picarro G2401 analyzer measures carbon monoxide (CO), carbon dioxide (CO2), methane (CH4), and water vapor (H2O) simultaneously in a single instrument.
            
            Key Specifications:
            - Detection limits: CO: 0.5 ppb, CO2: 0.1 ppm, CH4: 0.5 ppb, H2O: 0.01%
            - Precision: CO: 0.1 ppb, CO2: 0.02 ppm, CH4: 0.1 ppb
            - Measurement rate: 1 Hz
            - Warm-up time: < 30 minutes
            - Operating temperature: 5-40°C
            - Power consumption: 150W typical
            
            This analyzer is particularly useful for air quality monitoring, urban pollution studies, and combustion efficiency analysis.
            It features real-time data logging, automatic calibration, and can operate continuously for extended periods with minimal maintenance.
            
            The G2401 is widely deployed in urban air quality networks, industrial facilities, and research institutions worldwide.
            """,
            "metadata": {
                "product": "G2401",
                "category": "gas_analyzer",
                "technology": "CRDS",
                "applications": "air_quality_monitoring,urban_pollution_studies,combustion_analysis",
                "measurements": "CO,CO2,CH4,H2O",
                "detection_limits": "ppb_level",
                "type": "specifications"
            }
        },
        {
            "id": "g2131i_specs",
            "content": """
            Picarro's G2131-i isotopic analyzer provides high-precision measurements of stable isotopes in carbon dioxide and water vapor.
            
            Key Specifications:
            - δ13C precision: < 0.1‰
            - δ18O precision: < 0.1‰
            - δD precision: < 0.5‰
            - Measurement rate: 0.5 Hz
            - Warm-up time: < 2 hours
            - Operating temperature: 15-30°C
            - Power consumption: 200W typical
            
            The G2131-i uses wavelength-scanned cavity ring-down spectroscopy (WS-CRDS) to achieve isotopic precision that rivals traditional isotope ratio mass spectrometry (IRMS).
            It features automatic water vapor correction, real-time isotopic data processing, and continuous operation capabilities.
            
            These instruments are essential for understanding carbon cycling, ecosystem studies, climate research, and paleoclimatology.
            The isotopic data helps scientists trace the sources and sinks of greenhouse gases and understand biogeochemical processes.
            """,
            "metadata": {
                "product": "G2131-i",
                "category": "isotopic_analyzer",
                "technology": "WS-CRDS",
                "applications": "carbon_cycling,ecosystem_studies,climate_research,paleoclimatology",
                "measurements": "δ13C,δ18O,δD",
                "precision": "sub_per_mil",
                "type": "specifications"
            }
        },
        {
            "id": "crds_technology",
            "content": """
            Cavity Ring-Down Spectroscopy (CRDS) is the core technology behind Picarro's analyzers, providing unprecedented precision and stability in gas concentration measurements.
            
            How CRDS Works:
            1. Light from a laser is injected into a high-finesse optical cavity
            2. The cavity contains mirrors with reflectivity > 99.99%
            3. Light bounces back and forth thousands of times, creating a long effective path length
            4. When the laser is turned off, the light intensity decays exponentially
            5. The decay time (ring-down time) is measured with high precision
            6. Gas molecules in the cavity absorb light at specific wavelengths
            7. The presence of target gases shortens the ring-down time
            8. Concentration is calculated from the change in ring-down time
            
            Advantages of CRDS:
            - Superior stability: No baseline drift over months
            - High sensitivity: Parts-per-trillion detection limits possible
            - Absolute measurements: No calibration gases required
            - Wide dynamic range: 6+ orders of magnitude
            - Fast response: Sub-second measurements
            - Rugged design: Works in harsh environments
            
            Picarro's patented CRDS technology has revolutionized atmospheric monitoring and enabled new scientific discoveries in climate research, air quality assessment, and industrial process control.
            """,
            "metadata": {
                "category": "technology",
                "technology": "CRDS",
                "description": "core_technology_overview",
                "advantages": "stability,sensitivity,absolute_measurements",
                "applications": "atmospheric_monitoring,climate_research,industrial_control",
                "type": "technology_overview"
            }
        },
        {
            "id": "picarro_applications",
            "content": """
            Picarro analyzers are deployed worldwide in diverse applications across multiple industries and research fields.
            
            Environmental Monitoring:
            - Global greenhouse gas monitoring networks (NOAA, WMO)
            - Urban air quality assessment and compliance
            - Industrial emissions monitoring and reporting
            - Agricultural greenhouse gas flux measurements
            - Forest carbon sequestration studies
            
            Scientific Research:
            - Climate change research and modeling
            - Atmospheric chemistry studies
            - Ecosystem carbon cycling research
            - Paleoclimatology and ice core analysis
            - Ocean-atmosphere exchange studies
            
            Industrial Applications:
            - Natural gas leak detection and quantification
            - Industrial process control and optimization
            - Semiconductor manufacturing process monitoring
            - Pharmaceutical production quality control
            - Food and beverage industry applications
            
            Agricultural Research:
            - Crop yield optimization studies
            - Soil carbon sequestration measurements
            - Livestock methane emissions research
            - Precision agriculture applications
            - Carbon credit verification
            
            The instruments are known for their reliability in harsh environmental conditions, ability to provide continuous unattended operation for months at a time, and minimal maintenance requirements.
            """,
            "metadata": {
                "category": "applications",
                "description": "deployment_overview",
                "sectors": "environmental,scientific,industrial,agricultural",
                "reliability": "harsh_conditions,continuous_operation",
                "maintenance": "minimal",
                "type": "applications_overview"
            }
        },
        {
            "id": "reliability_features",
            "content": """
            Picarro analyzers are engineered for exceptional reliability and long-term stability, making them ideal for continuous monitoring applications.
            
            Hardware Reliability Features:
            - Rugged aluminum housing with IP65 protection
            - Temperature-controlled optical cavity for stability
            - Hermetically sealed optical components
            - Vibration-resistant design for field deployment
            - Redundant critical components where possible
            - Industrial-grade connectors and cabling
            
            Software and Control Features:
            - Automatic temperature and pressure compensation
            - Real-time data validation and quality control
            - Automatic calibration and drift correction
            - Remote monitoring and diagnostics capabilities
            - Comprehensive error logging and reporting
            - Fail-safe operation modes
            
            Environmental Adaptability:
            - Operating temperature range: -10°C to +45°C
            - Humidity tolerance: 0-95% non-condensing
            - Altitude capability: 0-4000m above sea level
            - Power supply flexibility: 100-240V AC, 50-60Hz
            - Low power consumption for solar/battery operation
            
            Maintenance and Service:
            - Mean time between failures (MTBF): > 50,000 hours
            - Predictive maintenance algorithms
            - Remote troubleshooting capabilities
            - Modular design for easy component replacement
            - Comprehensive service documentation and support
            
            These reliability features enable Picarro analyzers to operate continuously in remote locations, harsh environments, and critical applications where downtime is not acceptable.
            """,
            "metadata": {
                "category": "reliability",
                "description": "reliability_features",
                "hardware": "rugged_design,hermetic_sealing",
                "software": "automatic_compensation,remote_monitoring",
                "environmental": "wide_temperature_range,humidity_tolerant",
                "maintenance": "high_mtbf,predictive_maintenance",
                "type": "reliability_overview"
            }
        },
        {
            "id": "isotopic_measurements",
            "content": """
            Isotopic measurements are crucial for understanding biogeochemical cycles, climate processes, and environmental changes.
            
            What are Stable Isotopes:
            Stable isotopes are naturally occurring variants of elements that have the same number of protons but different numbers of neutrons. 
            Common stable isotopes measured include:
            - Carbon-13 (13C) and Carbon-12 (12C) in CO2
            - Oxygen-18 (18O) and Oxygen-16 (16O) in H2O
            - Deuterium (2H or D) and Hydrogen-1 (1H) in H2O
            
            Measurement Principles:
            - Isotopic ratios are expressed as delta (δ) values in per mil (‰)
            - δ13C = [(13C/12C)sample / (13C/12C)standard - 1] × 1000
            - δ18O = [(18O/16O)sample / (18O/16O)standard - 1] × 1000
            - δD = [(2H/1H)sample / (2H/1H)standard - 1] × 1000
            
            Applications of Isotopic Measurements:
            - Carbon cycle studies: Tracing CO2 sources and sinks
            - Water cycle research: Understanding precipitation patterns
            - Ecosystem studies: Plant water use efficiency
            - Climate reconstruction: Paleoclimate analysis
            - Food authentication: Origin verification
            - Medical diagnostics: Metabolic studies
            
            Picarro's isotopic analyzers use wavelength-scanned cavity ring-down spectroscopy (WS-CRDS) to achieve precision that rivals traditional isotope ratio mass spectrometry (IRMS) while providing continuous, real-time measurements in the field.
            """,
            "metadata": {
                "category": "isotopic_measurements",
                "description": "isotopic_principles",
                "isotopes": "13C,18O,2H",
                "applications": "carbon_cycle,water_cycle,ecosystem_studies",
                "technology": "WS-CRDS",
                "precision": "sub_per_mil",
                "type": "isotopic_overview"
            }
        },
        {
            "id": "data_quality",
            "content": """
            Picarro analyzers provide exceptional data quality through advanced measurement techniques and comprehensive quality control systems.
            
            Data Quality Features:
            - High precision: Sub-ppb detection limits for most gases
            - Excellent accuracy: < 1% uncertainty for most measurements
            - Long-term stability: < 0.1% drift per month
            - Fast response: Sub-second measurement rates
            - Wide dynamic range: 6+ orders of magnitude
            - Automatic calibration: Self-correcting measurements
            
            Quality Control Systems:
            - Real-time data validation algorithms
            - Automatic outlier detection and flagging
            - Comprehensive error logging and reporting
            - Performance monitoring and trending
            - Calibration verification procedures
            - Data completeness and consistency checks
            
            Data Output and Integration:
            - Standard data formats (ASCII, CSV, JSON)
            - Real-time data streaming capabilities
            - Network connectivity (Ethernet, WiFi, cellular)
            - Integration with SCADA and control systems
            - Cloud-based data storage and analysis
            - API access for custom applications
            
            The combination of advanced measurement technology and robust quality control ensures that Picarro analyzers provide reliable, high-quality data for critical applications in research, monitoring, and industrial control.
            """,
            "metadata": {
                "category": "data_quality",
                "description": "quality_features",
                "precision": "sub_ppb",
                "accuracy": "less_than_1_percent",
                "stability": "low_drift",
                "integration": "network_connectivity,api_access",
                "type": "quality_overview"
            }
        }
    ]


def check_document_exists(doc_processor: DocumentProcessor, content: str, metadata: Dict[str, Any]) -> bool:
    """
    Check if a document with similar content already exists in the knowledge base.
    
    Args:
        doc_processor: DocumentProcessor instance
        content: Document content to check
        metadata: Document metadata
        
    Returns:
        True if document exists, False otherwise
    """
    try:
        # Search for documents with similar content
        search_results = doc_processor.search_documents(content[:100], max_results=10)
        
        for result in search_results:
            # Check if the content is very similar (indicating same document)
            if result['content'][:200] == content[:200]:
                return True
                
        return False
    except Exception as e:
        logger.warning(f"Error checking document existence: {e}")
        return False


def load_demo_data():
    """
    Load demo documents into the knowledge base.
    """
    print("=== Loading Demo Data for Picarro SearchAI ===")
    
    try:
        # Initialize document processor
        doc_processor = DocumentProcessor()
        ai_responder = AIResponder(doc_processor)
        
        # Get demo documents
        demo_documents = get_demo_documents()
        
        print(f"Found {len(demo_documents)} demo documents to process...")
        
        added_count = 0
        skipped_count = 0
        
        for doc in demo_documents:
            print(f"\nProcessing: {doc['id']}")
            
            # Check if document already exists
            if check_document_exists(doc_processor, doc['content'], doc['metadata']):
                print(f"  ✓ Document already exists, skipping...")
                skipped_count += 1
                continue
            
            try:
                # Add document to knowledge base
                doc_id = ai_responder.add_document(doc['content'], doc['metadata'])
                print(f"  ✓ Added document: {doc_id}")
                print(f"    Category: {doc['metadata'].get('category', 'N/A')}")
                print(f"    Product: {doc['metadata'].get('product', 'N/A')}")
                added_count += 1
                
            except Exception as e:
                print(f"  ✗ Error adding document: {e}")
                logger.error(f"Error adding document {doc['id']}: {e}")
        
        # Print summary
        print(f"\n=== Summary ===")
        print(f"Documents added: {added_count}")
        print(f"Documents skipped (already exist): {skipped_count}")
        print(f"Total documents processed: {len(demo_documents)}")
        
        # Get knowledge base statistics
        print(f"\n=== Knowledge Base Statistics ===")
        stats = ai_responder.get_knowledge_base_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nDemo data loading completed successfully!")
        print(f"You can now test the system with queries about Picarro analyzers.")
        
    except Exception as e:
        print(f"Error loading demo data: {e}")
        logger.error(f"Demo data loading failed: {e}")
        raise


if __name__ == "__main__":
    load_demo_data() 