"""
SECONDO         (   )
  SecondoInterface    SECONDO      JDBC
"""

import subprocess
import logging
import tempfile
import os
import time
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import json


class SecondoConnector:
    """SECONDO         (       )"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 1234,
                 database: str = "berlintest",
                 user: str = "",
                 password: str = "",
                 SECONDO_bin_path: str = ""):
        """
           SECONDO   
        
        Args:
            host: SECONDO       
            port: SECONDO     
            database:      
            user:     (    )
            password:    (    )
            SECONDO_bin_path: SECONDO       
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.SECONDO_bin_path = SECONDO_bin_path
        
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
        
        self.query_timeout = 30
        
    def connect(self) -> bool:
        """
               
        
        Returns:
                  
        """
        try:
            
            if not os.path.exists(self.SECONDO_bin_path):
                self.logger.error(f"SECONDO        : {self.SECONDO_bin_path}")
                return False
            
            
            test_command = f"open database {self.database};\nquit;"
            result = self._execute_SECONDO_command(test_command, timeout=10)
            
            if result["success"]:
                self.connected = True
                self.logger.info(f"     SECONDO   : {self.host}:{self.port}/{self.database}")
                return True
            else:
                self.logger.error(f"  SECONDO     : {result['error']}")
                return False
            
        except Exception as e:
            self.logger.error(f"  SECONDO      : {e}")
            return False
    
    def disconnect(self):
        """       """
        self.connected = False
        self.logger.info("   SECONDO     ")
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
          SECONDO     -            
        
        Args:
            query: SECONDO    
            
        Returns:
                  
        """
        try:
            
            full_command = f"open database {self.database};\n{query};\nquit;"
            
            self.logger.debug(f"  SEGUNDO  : {query}")
            result = self._execute_SECONDO_command(full_command, timeout=self.query_timeout)
            
            if result["success"]:
                
                return self._parse_query_result(result["output"])
            else:
                self.logger.error(f"      : {result['error']}")
                return []
            
        except Exception as e:
            self.logger.error(f"       : {e}")
            self.logger.error(f"    : {query}")
            return []
    
    def execute_query_raw(self, query: str) -> str:
        """
          SECONDO               test_trains_id1.py      
        
        Args:
            query: SEGUNDO    
            
        Returns:
                       
        """
        try:
            
            full_command = f"open database {self.database};\nquery {query};\nquit;"
            
            self.logger.debug(f"  SEGUNDO  (    ): {query}")
            result = self._execute_SECONDO_command(full_command, timeout=self.query_timeout)
            
            if result["success"]:
                
                raw_data = self._extract_raw_data_section(result["output"])
                return raw_data if raw_data else "       "
            else:
                self.logger.error(f"      : {result['error']}")
                return f"      : {result['error']}"
            
        except Exception as e:
            self.logger.error(f"       : {e}")
            return f"    : {e}"
    
    def execute_command(self, command: str) -> bool:
        """
          SECONDO       -            
        
        Args:
            command: SECONDO  
            
        Returns:
                  
        """
        try:
            full_command = f"open database {self.database};\n{command};\nquit;"
            result = self._execute_SECONDO_command(full_command, timeout=self.query_timeout)
            
            if result["success"]:
                self.logger.debug(f"    SECONDO  : {command}")
                return True
            else:
                self.logger.error(f"      : {result['error']}")
                return False
            
        except Exception as e:
            self.logger.error(f"       : {e}")
            self.logger.error(f"  : {command}")
            return False
    
    def test_connection(self) -> bool:
        """
               
        
        Returns:
                  
        """
        try:
            
            test_query = "list objects;"
            result = self.execute_query(test_query)
            return len(result) > 0  
            
        except Exception as e:
            self.logger.error(f"      : {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
               
        
        Returns:
                   
        """
        info = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "connection_status": "connected" if self.connected else "disconnected",
            "connection_type": "command_line_interface"
        }
        
        return info
    
    def _execute_SECONDO_command(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """
          SECONDO     
        
        Args:
            command:     SECONDO  
            timeout:        
            
        Returns:
                  
        """
        try:
            
            result = subprocess.run(
                [self.SECONDO_bin_path],
                input=command,
                text=True,
                capture_output=True,
                timeout=timeout
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr,
                "returncode": result.returncode
            }
                
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "output": "",
                "error": "      ",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "returncode": -1
            }
    
    def _extract_raw_data_section(self, secondo_output: str) -> str:
        """ SEGUNDO             Id   Total runtime    """
        if not secondo_output:
            return ""
        
        lines = secondo_output.split('\n')
        data_lines = []
        in_data_section = False
        bracket_count = 0
        found_value = False
        
        for i, line in enumerate(lines):
            
            
            if line.strip().startswith('Id :') or line.strip().startswith('Id:'):
                in_data_section = True
            
            elif 'Total runtime' in line:
                in_data_section = True
                
                continue
            
            
            if in_data_section:
                
                if 'quit' in line.lower() or 'secondo =>' in line:
                    break
                
                
                if not line.strip():
                    continue
                
                
                if any(keyword in line for keyword in [
                    '=> []', 'command', 'started at:', 'noMemoryOperators', 'perOperator',
                    'Secondo =>', "'query", 'query Trains'
                ]):
                    continue
                    
                data_lines.append(line)
                
                
                if 'Value:' in line:
                    found_value = True
                    continue
                
                
                if found_value:
                    
                    open_brackets = line.count('(')
                    close_brackets = line.count(')')
                    bracket_count += open_brackets - close_brackets
                    
                    
                    if bracket_count <= 0 and line.strip().endswith(')))'):
                        break
                
                elif not found_value and line.strip().endswith(')))') and bracket_count <= 0:
                    break
        
        return '\n'.join(data_lines) if data_lines else ""

    def _parse_query_result(self, output: str) -> List[Dict[str, Any]]:
        """
          SEGUNDO              
        
        Args:
            output: SEGUNDO    
            
        Returns:
                    
        """
        try:
            
            raw_data = self._extract_raw_data_section(output)
            
            
            if raw_data:
                return [{"raw_data": raw_data}]
            
            
            lines = output.split('\n')
            object_list = []
            object_section = False
            
            for line in lines:
                line = line.strip()
                if 'short list' in line:
                    object_section = True
                    continue
                if object_section and line.startswith('*'):
                    
                    obj_name = line.replace('*', '').strip()
                    if obj_name:
                        object_list.append(obj_name)
                elif object_section and ('SECONDO =>' in line or line == ''):
                    break
            
            
            if object_list:
                return [{"object_name": obj_name} for obj_name in object_list]
            
            
            return [{
                "status": "executed", 
                "note": "Query executed successfully",
                "contains_trains": "Trains" in output,
                "contains_objects": "short list" in output
            }]
            
        except Exception as e:
            self.logger.error(f"         : {e}")
            return [{
                "error": str(e),
                "raw_output_preview": output[:500] + "..." if len(output) > 500 else output
            }]


class SecondoQueryBuilder:
    """SECONDO       """
    
    @staticmethod
    def build_trajectory_query(object_id: str = None, 
                              start_time: str = None, 
                              end_time: str = None,
                              location: str = None) -> str:
        """
                
        
        Args:
            object_id:   ID
            start_time:     
            end_time:     
            location:   
            
        Returns:
            SECONDO    
        """
        
        base_query = "query Trains feed"
        
        conditions = []
        
        
        if object_id:
            conditions.append(f"filter[.Id = {object_id}]")
        
        
        if start_time and end_time:
            time_period = f'[const periods value (("{start_time}" "{end_time}" TRUE TRUE))]'
            conditions.append(f"filter[.Trip present {time_period}]")
        
        
        if location:
            conditions.append(f"filter[.Trip passes {location}]")
        
        
        query = base_query
        for condition in conditions:
            query += f" {condition}"
        
        query += " consume"  
        
        return query
    
    @staticmethod
    def build_time_range_query(object_id: str = None, object_type: str = None) -> str:
        """
                
        
        Args:
            object_id:   ID
            object_type:      'train', 'taxi', 'car', 'bus'  
            
        Returns:
            SECONDO    
        """
        
        if object_type:
            if object_type.lower() == 'train':
                table_name = 'Trains'
            elif object_type.lower() == 'taxi':
                table_name = 'Taxis'
            elif object_type.lower() == 'car':
                table_name = 'Cars'
            elif object_type.lower() == 'bus':
                table_name = 'Buses'
            else:
                table_name = object_type.capitalize() + 's'
        else:
            table_name = 'Trains'
        
        query = f"query {table_name} feed"
        
        if object_id:
            query += f" filter[.Id = {object_id}]"
        
        
        query += " extend[StartTime: initial(.Trip), EndTime: final(.Trip)] project[Id, StartTime, EndTime] consume"
        
        return query
    
    @staticmethod
    def build_objects_list_query(object_type: str = None) -> str:
        """
                
        
        Args:
            object_type:      'train', 'taxi', 'car', 'bus'  
            
        Returns:
            SECONDO    
        """
        
        if object_type:
            
            if object_type.lower() == 'train':
                table_name = 'Trains'
            elif object_type.lower() == 'taxi':
                table_name = 'Taxis'
            elif object_type.lower() == 'car':
                table_name = 'Cars'
            elif object_type.lower() == 'bus':
                table_name = 'Buses'
            else:
                
                table_name = object_type.capitalize() + 's'
        else:
            
            table_name = 'Trains'
        
        query = f"query {table_name} feed project[Id] consume"
        
        return query
    
    @staticmethod
    def build_locations_query(object_id: str = None, places_csv_path: str = None) -> List[str]:
        """
               -             
        
        Args:
            object_id:   ID
            places_csv_path: places.csv        knowledge_base/places.csv
            
        Returns:
            SECONDO      
        """
        import csv
        import os
        
        queries = []
        
        
        if places_csv_path is None:
            places_csv_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "places.csv")
        
        if object_id:
            try:
                
                with open(places_csv_path, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        place_name = row['name'].strip()  
                        if place_name:  
                            
                            query = f"query Trains feed filter[.Id = {object_id}] filter[.Trip passes {place_name}] consume"
                            queries.append(query)
            except FileNotFoundError:
                print(f"  :     places.csv  : {places_csv_path}")
                
                query = f"query Trains feed filter[.Id = {object_id}] filter[.Trip passes xxx] consume"
                queries.append(query)
            except Exception as e:
                print(f"  places.csv   : {e}")
                
                query = f"query Trains feed filter[.Id = {object_id}] filter[.Trip passes xxx] consume"
                queries.append(query)
        else:
            
            query = "query Places feed project[Name] consume"
            queries.append(query)
        
        return queries



def main():
    """     """
    
    connector = SecondoConnector(
        host="localhost",
        port=1234,
        database="berlintest",
        SECONDO_bin_path=""
    )
    
    
    if connector.test_connection():
        print("  SECONDO         ")
        
        
        info = connector.get_database_info()
        print(f"     : {info}")
        
        
        query = SecondoQueryBuilder.build_trajectory_query(object_id="1")
        print(f"    : {query}")
        
        
        try:
            results = connector.execute_query(query)
            print(f"      : {len(results)}")
            if results:
                print("      :")
                for i, result in enumerate(results):
                    print(f"   {i+1}: {result}")
            else:
                print("       ")
        except Exception as e:
            print(f"      : {e}")
        
    else:
        print("  SECONDO         ")
        print("    ")
        print("1. SECONDO       ")
        print("2. SECONDO         ")
        print("3.        ")
    
    
    connector.disconnect()


if __name__ == "__main__":
    main()
