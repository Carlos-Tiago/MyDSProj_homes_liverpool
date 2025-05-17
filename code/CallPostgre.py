import psycopg2 as psycopg2
import yaml
from psycopg2 import sql #redundant but for clarity reasons
from psycopg2.extensions import AsIs
import psycopg2.extras
import sys
import pandas as pd
import re

# Went with a class to keep track of the state of the connection/cursor variables
# No point having more than 1 object for the scope of this project
class Database:
    
    connection = None
    cursor = None
    status = None
    
    def __init__(self):
        # Connect #
        #
        # Establish only 1 connection at a time with the db
        if Database.status is None or "disconnected" : 
            # Load data from the secrets file
            # 'with' ensures automatic acquisition and release of resources
            with open('db_credentials.yml', 'r') as file: 
                file_credentials = yaml.safe_load(file)

            # Connect to the database
            credentials_dbhomes = file_credentials['homes credentials']
            try:
                Database.connection = psycopg2.connect(
                    host=credentials_dbhomes['host'],
                    database=credentials_dbhomes['database'],
                    user=credentials_dbhomes['user'],
                    password=credentials_dbhomes['password'])
                # Automatically submit to the database the changes done by queries. 
                Database.connection.autocommit = True 
                Database.status = "connected"
            finally:
                print(f"Connection status: {Database.status}")
        print(f"\n-------------------------------------------------"
              f"\nSuccessfully connected to the database: {credentials_dbhomes['database']}"
              f"\n-------------------------------------------------")
        Database.cursor = Database.connection.cursor()                  
        Database.cursor.execute('SELECT version()')
        print(Database.cursor.fetchone())
        

    # Close connection
    @classmethod # As there'll be no instances
    def disconnect(cls):
        Database.cursor.close()
        Database.connection.close()
        Database.status = "disconnected"
        print(f"Database connection closed.")
        
        
    @classmethod
    def createTableRealEstateAgent(cls):
        query= ("""
        CREATE TABLE real_estate_agent(
            real_estate_agent_id SERIAL PRIMARY KEY, 
            name VARCHAR(255) NOT NULL UNIQUE,
            telephone VARCHAR(255) NOT NULL UNIQUE
            /*Each real estate agent will be added as data is collected (names too unrestrited to be predictable)*/
        );
        """)
        Database.cursor.execute(query)
        
        
    @classmethod
    def createTableHomeLiverpool(cls):
        query = ("""
        CREATE TABLE home_liverpool(
            home_liverpool_id INT PRIMARY KEY,   /*(docs) - Quoting identifiers ("ID", "Price") is only necessary if we want to keep the upper casing. Non-quoted identifiers (ID , Price) get automatically folded to lower case (id, price) and registered as such in the query. */
            price INT CHECK(price > 0) NOT NULL,
            is_auction BOOLEAN NOT NULL,
            is_shared_ownership BOOLEAN NOT NULL,
            title VARCHAR(255) NOT NULL,
            location VARCHAR(255) NOT NULL,
            coordinate_x DOUBLE PRECISION, /* same as FLOAT(25-53) */
            coordinate_y DOUBLE PRECISION,
            postal_code VARCHAR(10) NOT NULL, 
            bedroom_count INT,
            bathroom_count INT,
            living_room_count INT,
            property_size INT, /* sq. feet */
            epc_rating CHAR(1), 
            feature_set TEXT[],
            listing_date DATE NOT NULL,
            photos TEXT[],
            description TEXT,
            tenure VARCHAR(20),
            time_remaining_on_lease_years INT CHECK(time_remaining_on_lease_years > 0),
            annual_service_charge INT CHECK(annual_service_charge >= 0), /*monetary values given in pounds(£) */
            council_tax_band CHAR(1), 
            ground_rent INT CHECK(ground_rent >= 0), 
            real_estate_agent_id INT REFERENCES real_estate_agent ON DELETE SET NULL ON UPDATE CASCADE /* SET NULL...> information about the home such as adress and specifications can still be relevant */
        );
        """)
        Database.cursor.execute(query)
    
    
    @classmethod
    def insert(cls, home_liverpool_id, price, is_auction, is_shared_ownership, 
               title, location, coordinate_x, coordinate_y, postal_code, 
               bedroom_count, bathroom_count, living_room_count, property_size,
               epc_rating, feature_set, listing_date, photos, description, tenure, 
               time_remaining_on_lease_years, annual_service_charge, 
               council_tax_band, ground_rent, real_estate_agent_name, 
               real_estate_agent_telephone):

        # 'real_estate_agent' table #
        query = ("""
            INSERT INTO real_estate_agent(name, telephone) 
            VALUES (%s, %s) 
            ON CONFLICT (name) DO UPDATE
            SET telephone = EXCLUDED.telephone; /* In case the R.E contact gets updated */
        """)
        # Query parameter must be a tuple such as [bar] or (bar,). Having (bar)
        #  won't be iterable and thus not work.
        Database.cursor.execute(query, 
                                (real_estate_agent_name, real_estate_agent_telephone)) 
        
        # GIVE UPDATED ID AS PARAMETER TO THE INSERT BELOW #
        #
        # Get updated r.e.a. serial id (to link with the current home)
        query = ("""
            SELECT rea.real_estate_agent_id 
            FROM real_estate_agent rea
            WHERE rea.name = %s
        
        """)
        Database.cursor.execute(query, (real_estate_agent_name,))
        updated_rea_id = Database.cursor.fetchone() 
        print(f"\nupdated id: {updated_rea_id} for agent {real_estate_agent_name}")
        
        # 'home_liverpool' table part #
        query =("""
            INSERT INTO home_liverpool (home_liverpool_id, price, is_auction, is_shared_ownership, title, location, coordinate_x, coordinate_y, postal_code, bedroom_count, bathroom_count, living_room_count, property_size, epc_rating, feature_set, listing_date, photos, description, tenure, time_remaining_on_lease_years, annual_service_charge, council_tax_band, ground_rent, real_estate_agent_id) 
            VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (home_liverpool_id) DO UPDATE /* in case a house price gets updated */
            SET price = EXCLUDED.price;
            """)
        Database.cursor.execute(query, 
                                (home_liverpool_id, price, is_auction,
                                 is_shared_ownership, title, location, coordinate_x,
                                 coordinate_y, postal_code, bedroom_count,
                                 bathroom_count, living_room_count, property_size,
                                 epc_rating, feature_set, listing_date, photos,
                                 description, tenure, time_remaining_on_lease_years,
                                 annual_service_charge, council_tax_band, 
                                 ground_rent, updated_rea_id))
      
    
    @classmethod
    def replaceColumnConstraint(cls, target_table_name, target_column_name,
                                **kwargs):
        '''
        kwargs =  constraint_to_drop_is_not_null, constraint_to_add
        usage:
            ONLY ADD a constraint to a column: 
                -ommit <constraint_to_drop=...> from the arguments
            ONLY REMOVE a constraint from a column:
                -ommit <constraint_to_add=...> from the arguments
            REPLACE:
                -keep both.
            NOTE: The same constraint type (such as UNIQUE) can be added multiple
                  times to the same column in the current implementation by
                  accidentally running the function with the same parameters
                  twice. That doesn't seem to cause any issues (and can always be
                  manually deleted). Should anyone want to stop this from happening,
                  add an argument to this function such as <custom_constraint_name>
                  to make it mandatory to name them, and swap the query to
                  "ALTER TABLE <table> ADD CONSTRAINT <custom_constrain_name>
                  <SOME_CONSTRAINT> (<column>). This way a double run should
                  trigger an error instead.
            
            Functionality for the CHECK () NOT YET IMPLEMENTED.
        
        '''
        
        # No need for a try catch so that in case of a drop + and add, if the
        #  add fails (typo in constraint name), then the drop doesn't happen.
        # That error would be raised beforehand, during compiling(?), outside
        #  of the try-block regardless.
        if 'constraint_to_drop' in kwargs:
            # NOT NULL has a set name and a slightly different syntax
            # For other constraints we get their name first and drop them
            if kwargs['constraint_to_drop'].casefold() != "not null":
                query = ("""
                    SELECT constraint_name 
                    FROM information_schema.constraint_column_usage isccu
                    WHERE isccu.table_name = %s AND isccu.column_name = %s;
                """)
                Database.cursor.execute(query, (target_table_name, target_column_name))
                # Convert query result from tuple to str #
                #
                # *Will contain 1 element
                constraint_name_tuple = Database.cursor.fetchall() #*
                constraint_name_string = [("").join(constraint) for constraint in constraint_name_tuple][0] 

                # Drop the constraint of the name obtained
                query = sql.SQL("""
                    ALTER TABLE {} DROP CONSTRAINT {};
                """).format(sql.Identifier(target_table_name),
                            sql.Identifier(constraint_name_string))
                Database.cursor.execute(query)

            # Dropping NOT NULL
            elif kwargs['constraint_to_drop'].casefold() == "not null":
                query = sql.SQL("""
                    ALTER TABLE {} ALTER COLUMN {} DROP NOT NULL;
                """).format(sql.Identifier(target_table_name),
                            sql.Identifier(target_column_name))
                Database.cursor.execute(query)

                
        if 'constraint_to_add' in kwargs:
            if kwargs['constraint_to_add'].casefold() != "not null":
                query = sql.SQL("""
                    ALTER TABLE {table} ADD %s ({column});
                """).format(table=sql.Identifier(target_table_name),
                            column=sql.Identifier(target_column_name))
                Database.cursor.execute(query, (AsIs(kwargs['constraint_to_add']),))

            elif kwargs['constraint_to_add'].casefold() == "not null":
                query = sql.SQL("""
                    ALTER TABLE {} ALTER COLUMN {} SET NOT NULL;
                """).format(sql.Identifier(target_table_name),
                            sql.Identifier(target_column_name))
                Database.cursor.execute(query)
        
       
    @classmethod
    # kwargs = query, values, identifiers
    def custom_query(cls, query, **kwargs): 
        print(f"query: {query}")
        print(f"kw_values: {kwargs['kw_values']}")
        print(f"kw_identifiers {kwargs['kw_identifiers']}")
        
        Database.cursor.execute(query, kwargs['kw_values'])
                   


    @classmethod
    def postgresql_to_dataframe(cls, select_query, column_names):
        Database.cursor.execute(select_query)
        data = Database.cursor.fetchall()
        df = pd.DataFrame(data, columns=column_names)
        
        return df
  

    @classmethod
    def dataframe_to_postgresql(cls, df_cleaned, table_name):
        print(f"Table:\n{table_name}\n")
        cleaned_table_name = table_name + "_cleaned"
        #Remove any pre-existing cleaned table
        Database.cursor.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(cleaned_table_name)))
        
        
        # NOTE: THE COLUMNS TO CREATE WILL DEPEND ON THE TABLE AND COLUMNS THAT
        #       DIDN'T PASS THE MISSING VALUE THRESHOLD
        #!# MAY REQUIRE EDIT
        #
        # For home tables of format: 'home_something_something_etc'
        # *ready for the possibility of adding tables of other cities in the future
        regex_home_tables = re.compile("^home_[a-z]+(_[a-z]+)*$") #*
        
        # For the table(s?) of format: 'real_estate_agent_etc' 
        regex_real_estate_agent_tables = re.compile("^real_estate_agent(_[a-z]+)*$")
        
        #Create tables for the cleaned data if they don't exist
        if re.match(regex_real_estate_agent_tables,  table_name) is not None:
            # Optional step so that the cleaned table has the same column as
            #  the original one
            df_cleaned = df_cleaned.reindex(columns=['real_estate_agent_id',
                                                     'name', 'telephone'])
            
            create_query=sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}(
                real_estate_agent_id SERIAL PRIMARY KEY, 
                name VARCHAR(255) NOT NULL UNIQUE,
                telephone VARCHAR(255) NOT NULL UNIQUE
                );
            """).format(sql.Identifier(cleaned_table_name))
            Database.cursor.execute(create_query)
           
        elif re.match(regex_home_tables, table_name) is not None:
            df_cleaned = df_cleaned.reindex(columns=['home_liverpool_id', 'price',
                                                     'is_auction',
                                                     'is_shared_ownership',
                                                     'title', 'location',
                                                     'coordinate_x', 'coordinate_y',
                                                     'postal_code', 'bedroom_count',
                                                     'bathroom_count',
                                                     'living_room_count',
                                                     'feature_set', 'listing_date',
                                                     'photos', 'description',
                                                     'tenure', 
                                                     'real_estate_agent_id'])
            
            create_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {}(
                    home_liverpool_id INT PRIMARY KEY,
                    price INT CHECK(price > 0) NOT NULL,
                    is_auction BOOLEAN NOT NULL,
                    is_shared_ownership BOOLEAN NOT NULL,
                    title VARCHAR(255) NOT NULL,
                    location VARCHAR(255) NOT NULL,
                    coordinate_x DOUBLE PRECISION,
                    coordinate_y DOUBLE PRECISION,
                    postal_code VARCHAR(10) NOT NULL, 
                    bedroom_count INT,
                    bathroom_count INT,
                    living_room_count INT,
                    feature_set TEXT[],
                    listing_date DATE NOT NULL,
                    photos TEXT[],
                    description TEXT,
                    tenure VARCHAR(20),
                    real_estate_agent_id INT REFERENCES real_estate_agent_cleaned ON DELETE SET NULL ON UPDATE CASCADE /* SET NULL...> information about the home such as adress and specifications can still be relevant */
                );
            """).format(sql.Identifier(cleaned_table_name))
            Database.cursor.execute(create_query)
         
        else:
            print(f"WARNING: Table name posesses unusual name format!... ")

        
        # Insert on the new tables
        column_names = df_cleaned.columns.tolist()
        # *type casting column_names as an sql.Identifier
        insert_query = sql.SQL("""
            INSERT INTO {} ({})
            VALUES %s /* execute_values only takes one value, which is a double composite (a composite of tuples) */
        """).format(sql.Identifier(cleaned_table_name),
                    sql.SQL(', ').join(map(sql.Identifier, column_names)) #*
                   )
        # Exclude the itertuples index from being present so as not to add an
        #  extra column, also no "Pandas" name
        tuples = [tupl for tupl in df_cleaned.itertuples(index=None, name=None)] 
        psycopg2.extras.execute_values(Database.cursor, insert_query, tuples)





