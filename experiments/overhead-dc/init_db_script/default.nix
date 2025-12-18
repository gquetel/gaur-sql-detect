# Expression creating a .sql file to automate tables creation and population.
let
  inputs = import ../../../npins;
  pkgs = import inputs.nixpkgs_mysql_8_4_3 { };
  airportFiles = inputs.ourairports_data;
in

pkgs.writeTextFile {

  name = "init_db.sql";
  text = ''
    ALTER USER 'root' @'localhost' IDENTIFIED BY 'root';
    create database users;
    use users;
    CREATE USER 'toto' @'localhost' IDENTIFIED BY 'toto';
    GRANT 
    SELECT 
      ON dataset.* TO 'toto' @'localhost' WITH GRANT OPTION;
    flush privileges;
    create database dataset;
    use dataset;
    CREATE TABLE airport (
      id INT PRIMARY KEY AUTO_INCREMENT, 
      ident VARCHAR(50) NOT NULL UNIQUE, 
      type VARCHAR(20) NOT NULL, 
      name VARCHAR(255) NOT NULL, 
      latitude_deg DECIMAL(10, 6), 
      longitude_deg DECIMAL(10, 6), 
      elevation_ft INT, 
      continent CHAR(2), 
      iso_country CHAR(2), 
      iso_region VARCHAR(10), 
      municipality VARCHAR(100), 
      scheduled_service VARCHAR(3), 
      gps_code VARCHAR(10), 
      icao_code VARCHAR(10), 
      iata_code VARCHAR(5), 
      local_code VARCHAR(10), 
      home_link VARCHAR(255), 
      wikipedia_link VARCHAR(255), 
      keywords TEXT
    );
    LOAD DATA INFILE '${airportFiles}/airports.csv' INTO TABLE airport FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES (
      @v1, @v2, @v3, @v4, @v5, @v6, @v7, @v8, 
      @v9, @v10, @v11, @v12, @v13, @v14, @v15, 
      @v16, @v17, @v18, @v19
    ) 
    SET 
      id = NULLIF(@v1, ""), 
      ident = NULLIF(@v2, ""), 
      type = NULLIF(@v3, ""), 
      name = NULLIF(@v4, ""), 
      latitude_deg = NULLIF(@v5, ""), 
      longitude_deg = NULLIF(@v6, ""), 
      elevation_ft = NULLIF(@v7, ""), 
      continent = NULLIF(@v8, ""), 
      iso_country = NULLIF(@v9, ""), 
      iso_region = NULLIF(@v10, ""), 
      municipality = NULLIF(@v11, ""), 
      scheduled_service = NULLIF(@v12, ""), 
      gps_code = NULLIF(@v13, ""), 
      icao_code = NULLIF(@v14, ""), 
      iata_code = NULLIF(@v15, ""), 
      local_code = NULLIF(@v16, ""), 
      home_link = NULLIF(@v17, ""), 
      wikipedia_link = NULLIF(@v18, ""), 
      keywords = NULLIF(@v19, "");
    CREATE TABLE airport_frequencies (
      id INTEGER PRIMARY KEY, 
      airport_ref INTEGER, 
      airport_ident VARCHAR(50), 
      type VARCHAR(50), 
      description VARCHAR(255), 
      frequency_mhz DECIMAL(10, 3), 
      FOREIGN KEY (airport_ref) REFERENCES airport(id), 
      FOREIGN KEY (airport_ident) REFERENCES airport(ident)
    );
    LOAD DATA INFILE '${airportFiles}/airport-frequencies.csv' INTO TABLE airport_frequencies FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES (@v1, @v2, @v3, @v4, @v5, @v6) 
    SET 
      id = NULLIF(@v1, ""), 
      airport_ref = NULLIF(@v2, ""), 
      airport_ident = NULLIF(@v3, ""), 
      type = NULLIF(@v4, ""), 
      description = NULLIF(@v5, ""), 
      frequency_mhz = NULLIF(@v6, "");
    CREATE TABLE runways (
      id INT PRIMARY KEY AUTO_INCREMENT, 
      airport_ref INT NOT NULL, 
      airport_ident VARCHAR(10) NOT NULL, 
      length_ft INT, 
      width_ft INT, 
      surface VARCHAR(255), 
      lighted TINYINT(1), 
      closed TINYINT(1), 
      le_ident VARCHAR(10), 
      le_latitude_deg DECIMAL(10, 6), 
      le_longitude_deg DECIMAL(10, 6), 
      le_elevation_ft INT, 
      le_heading_degT DECIMAL(5, 1), 
      le_displaced_threshold_ft INT, 
      he_ident VARCHAR(10), 
      he_latitude_deg DECIMAL(10, 6), 
      he_longitude_deg DECIMAL(10, 6), 
      he_elevation_ft INT, 
      he_heading_degT DECIMAL(5, 1), 
      he_displaced_threshold_ft INT, 
      FOREIGN KEY (airport_ref) REFERENCES airport(id), 
      FOREIGN KEY (airport_ident) REFERENCES airport(ident)
    );
    LOAD DATA INFILE '${airportFiles}/runways.csv' INTO TABLE runways FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES (
      @v1, @v2, @v3, @v4, @v5, @v6, @v7, @v8, 
      @v9, @v10, @v11, @v12, @v13, @v14, @v15, 
      @v16, @v17, @v18, @v19, @v20
    ) 
    SET 
      id = NULLIF(@v1, ""), 
      airport_ref = NULLIF(@v2, ""), 
      airport_ident = NULLIF(@v3, ""), 
      length_ft = NULLIF(@v4, ""), 
      width_ft = NULLIF(@v5, ""), 
      surface = NULLIF(@v6, ""), 
      lighted = NULLIF(@v7, ""), 
      closed = NULLIF(@v8, ""), 
      le_ident = NULLIF(@v9, ""), 
      le_latitude_deg = NULLIF(@v10, ""), 
      le_longitude_deg = NULLIF(@v11, ""), 
      le_elevation_ft = NULLIF(@v12, ""), 
      le_heading_degT = NULLIF(@v13, ""), 
      le_displaced_threshold_ft = NULLIF(@v14, ""), 
      he_ident = NULLIF(@v15, ""), 
      he_latitude_deg = NULLIF(@v16, ""), 
      he_longitude_deg = NULLIF(@v17, ""), 
      he_elevation_ft = NULLIF(@v18, ""), 
      he_heading_degT = NULLIF(@v19, ""), 
      he_displaced_threshold_ft = NULLIF(@v20, "");
    CREATE TABLE navaids (
      id INTEGER PRIMARY KEY, 
      filename VARCHAR(255), 
      ident VARCHAR(50), 
      name VARCHAR(255), 
      type VARCHAR(50), 
      frequency_khz INTEGER, 
      latitude_deg DECIMAL(10, 6), 
      longitude_deg DECIMAL(10, 6), 
      elevation_ft INTEGER, 
      iso_country VARCHAR(10), 
      dme_frequency_khz INTEGER, 
      dme_channel VARCHAR(10), 
      dme_latitude_deg DECIMAL(10, 6), 
      dme_longitude_deg DECIMAL(10, 6), 
      dme_elevation_ft INTEGER, 
      slaved_variation_deg DECIMAL(7, 3), 
      magnetic_variation_deg DECIMAL(7, 3), 
      usageType VARCHAR(50), 
      power VARCHAR(50), 
      associated_airport VARCHAR(50), 
      FOREIGN KEY (associated_airport) REFERENCES airport(ident)
    );
    -- We don't care if some checks are not done.
    SET 
      FOREIGN_KEY_CHECKS = 0;
    LOAD DATA INFILE '${airportFiles}/navaids.csv' INTO TABLE navaids FIELDS TERMINATED BY ',' LINES TERMINATED BY '\n' IGNORE 1 LINES (
      @v1, @v2, @v3, @v4, @v5, @v6, @v7, @v8, 
      @v9, @v10, @v11, @v12, @v13, @v14, @v15, 
      @v16, @v17, @v18, @v19, @v20
    ) 
    SET 
      id = NULLIF(@v1, ""), 
      filename = NULLIF(@v2, ""), 
      ident = NULLIF(@v3, ""), 
      name = NULLIF(@v4, ""), 
      type = NULLIF(@v5, ""), 
      frequency_khz = NULLIF(@v6, ""), 
      latitude_deg = NULLIF(@v7, ""), 
      longitude_deg = NULLIF(@v8, ""), 
      elevation_ft = NULLIF(@v9, ""), 
      iso_country = NULLIF(@v10, ""), 
      dme_frequency_khz = NULLIF(@v11, ""), 
      dme_channel = NULLIF(@v12, ""), 
      dme_latitude_deg = NULLIF(@v13, ""), 
      dme_longitude_deg = NULLIF(@v14, ""), 
      dme_elevation_ft = NULLIF(@v15, ""), 
      slaved_variation_deg = NULLIF(@v16, ""), 
      magnetic_variation_deg = NULLIF(@v17, ""), 
      usageType = NULLIF(@v18, ""), 
      power = NULLIF(@v19, ""), 
      associated_airport = NULLIF(@v20, "");
    CREATE TABLE countries (
      id INTEGER PRIMARY KEY, 
      code VARCHAR(2) UNIQUE NOT NULL, 
      name VARCHAR(100), 
      continent VARCHAR(2), 
      wikipedia_link VARCHAR(500), 
      keywords TEXT
    );
    LOAD DATA INFILE '${airportFiles}/countries.csv' INTO TABLE countries FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES (@v1, @v2, @v3, @v4, @v5, @v6) 
    SET 
      id = NULLIF(@v1, ""), 
      code = NULLIF(@v2, ""), 
      name = NULLIF(@v3, ""), 
      continent = NULLIF(@v4, ""), 
      wikipedia_link = NULLIF(@v5, ""), 
      keywords = NULLIF(@v6, "");
    CREATE TABLE regions (
      id INTEGER PRIMARY KEY, 
      code VARCHAR(10) UNIQUE NOT NULL, 
      local_code VARCHAR(10), 
      name VARCHAR(100), 
      continent VARCHAR(2), 
      iso_country VARCHAR(2), 
      wikipedia_link VARCHAR(500), 
      keywords TEXT
    );
    LOAD DATA INFILE '${airportFiles}/regions.csv' INTO TABLE regions FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES (
      @v1, @v2, @v3, @v4, @v5, @v6, @v7, @v8
    ) 
    SET 
      id = NULLIF(@v1, ""), 
      code = NULLIF(@v2, ""), 
      local_code = NULLIF(@v3, ""), 
      name = NULLIF(@v4, ""), 
      continent = NULLIF(@v5, ""), 
      iso_country = NULLIF(@v6, ""), 
      wikipedia_link = NULLIF(@v7, ""), 
      keywords = NULLIF(@v8, "");
  '';
}
