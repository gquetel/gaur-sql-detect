import logging
import mysql
import mysql.connector

logger = logging.getLogger(__name__)


class SQLConnector:
    def __init__(self, user: str, pwd: str, socket_path: str, database: str):
        self.user = user
        self.pwd = pwd
        self.socket_path = socket_path
        self.database = database
        self.cnx = None

    def init_new_cnx(self):
        """Return a new connection to the MySQL server.

        Warning: Establishing a connection to the MySQL server will invoke
        the parser 3 times. The gaur.log file should be emptied accordingly
        when calling this function.

        Returns:
            mysql.connector.MySQLConnection: Connection to the MySQL server.
        """
        self.cnx = mysql.connector.connect(
            user=self.user,
            password=self.pwd,
            unix_socket=self.socket_path,
            database=self.database,
            read_timeout=10,
        )

    def execute_query(self, query: str) -> int:
        """Execute query and return its success.

        Args:
            query (_type_): _description_

        Returns:
            int: retcode, 0 when expected behavior, 1 when an unexpected error
            occured and the logfile shouldn't be used.
        """
        # https://dev.mysql.com/doc/connector-python/en/connector-python-multi.html

        with self.cnx.cursor(buffered=True) as cur:
            try:
                cur.execute(query)
                for _, _ in cur.fetchsets():
                    # We don't care about the results, we only want to collect the
                    # traces
                    pass

            except mysql.connector.Error as e:
                # We list the semantic errors: there was no error while parsing,
                # the data collector was able to generate logs
                errno_with_trace = [
                    1044,  # Access denied
                    1048,  # Column cannot be null
                    1049,  # Database unknown
                    1054,  # Unknown column
                    1062,  # Duplicate entry
                    1064,  # Syntax error
                    1093,  # Error with table specification.
                    1094,  # Unknown thread id
                    1105,  # XPATH Syntax error
                    1109,  # Unknown table
                    1136,  # Column count does not match
                    1140,  # Aggregation error
                    1142,  # Access denied
                    1146,  # Table unknown
                    1193,  # Unknown system variable
                    1222,  # Different number of columns betwen 2 selects
                    1227,  # Access denied
                    1241,  # Operand should contain only 1 column
                    1242,  # Subquery returns more than 1 row
                    1248,  # Every derived table must have its own alias
                    1264,  # Value out of range
                    1265,  # Data truncated for column
                    1288,  # The target table is not updatable
                    1292,  # Truncated incorrect double value
                    1305,  # Procedure/function unknown
                    1317,   # Query execution was interrupted
                    1327,  # Undeclared variable
                    1364,  # Field doesn't have a default value
                    1366,  # Incorrect integer value
                    1370,  # Routine execution denied
                    1406,  #  Data too long
                    1451,
                    1452,  # Cannot add or update a child row
                    1525,  # Incorrect DATETIME value
                    1582,  # Incorrect parameters
                    1584,  # Incorrect parameters
                    1630,  # Function does not exists.
                    1690,  # BIGINT Issue
                    1772,  # Malformed GTID set specification
                    2014,  # Multiple log entries
                    3024,  # The Read Operation timed out.
                    3141,  # Invalid JSON text
                    3691,  # Mismatched parenthesis
                    3980,  # Invalid JSON attribute
                    3995,  # Wrong Character set
                ]

                # Log errors, those appearing here are not normal, hence the
                # critical. We do not want to crash the collection so we only
                # Reset the logfile and return an empty DataFrame.
                if e.errno not in errno_with_trace:
                    logger.critical(f"Error for query {query} {e}, {e.errno}")
                    return 1
            return 0
